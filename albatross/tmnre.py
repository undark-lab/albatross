print(
    r"""
        A       Initialising albatross
    ___/_\___   ----------------------
     ',. ..'    Type: TMNRE
     /.'^'.\    Authors: J. Alvey, M. Gerdes
    /'     '\   Version: v0.0.1 | April 2023
"""
)

import sys
import numpy as np
from datetime import datetime
import glob
import pickle
import swyft.lightning as sl
import sstrax as st
from config_utils import read_config, init_config
from simulator_utils import init_simulator
from inference_utils import (
    save_logratios,
    setup_zarr_store,
    setup_dataloader,
    setup_trainer,
    init_network,
    update_bounds,
    save_logratios,
)

# For parallelisation
import subprocess
import psutil
import logging
import time


def main(args):
    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Reading config file"
    )
    # Load and parse config file
    tmnre_parser = read_config(args)
    conf = init_config(tmnre_parser, args)
    logging.basicConfig(
        filename=f"{conf['zarr_params']['store_path']}/log_{conf['zarr_params']['run_id']}.log",
        filemode="w",
        format="%(asctime)s | %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )
    simulator = init_simulator(
        st.simulate_stream, st.halo_to_gd1_vmap, st.halo_to_gd1_velocity_vmap, conf
    )
    bounds = None
    if conf["tmnre"]["generate_obs"]:
        obs = simulator.generate_observation()
        logging.warning(
            f"Overwriting observation file: {conf['zarr_params']['store_path']}/observation_{conf['zarr_params']['run_id']}"
        )
        with open(
            f"{conf['zarr_params']['store_path']}/observation_{conf['zarr_params']['run_id']}",
            "wb",
        ) as f:
            pickle.dump(obs, f)
    else:
        observation_path = conf["tmnre"]["obs_path"]
        with open(observation_path, "rb") as f:
            obs = pickle.load(f)
        subprocess.run(
            f"cp {observation_path} {conf['zarr_params']['store_path']}/observation_{conf['zarr_params']['run_id']}",
            shell=True,
        )
    logging.info(
        f"Observation loaded and saved in {conf['zarr_params']['store_path']}/observation_{conf['zarr_params']['run_id']}"
    )
    for round_id in range(1, int(conf["tmnre"]["num_rounds"]) + 1):
        start_time = datetime.now()
        # Initialise the simulator, including a stream generator of choice
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Initialising simulator"
        )
        simulator = init_simulator(
            st.simulate_stream, st.halo_to_gd1_vmap, st.halo_to_gd1_velocity_vmap, conf
        )
        # Initialise the zarr store to save the simulations
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Initialising zarrstore for round {round_id}"
        )
        store = setup_zarr_store(conf, simulator, round_id=round_id)
        # Simulate the first round of simulations
        logging.info(f"Starting simulations for round {round_id}")
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Simulating data for round {round_id}"
        )
        if conf["zarr_params"]["njobs"] == -1:
            njobs = psutil.cpu_count(logical=True)
        elif conf["zarr_params"]["njobs"] > psutil.cpu_count(logical=False):
            njobs = psutil.cpu_count(logical=True)
        elif conf["zarr_params"]["run_parallel"]:
            njobs = conf["zarr_params"]["njobs"]
        else:
            njobs = 1
        while store.sims_required > 0:
            processes = []
            print(
                f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | (Re)starting {njobs} simulation batches. {store.sims_required} simulations still required"
            )
            for job in range(njobs):
                p = subprocess.Popen(
                    [
                        "python",
                        "run_parallel.py",
                        f"{conf['zarr_params']['store_path']}/config_{conf['zarr_params']['run_id']}.txt",
                        str(round_id),
                    ]
                )
                processes.append(p)
            status_array = np.array([None for p in processes])
            while np.all(status_array == None) and len(status_array) != 0:
                time.sleep(60)
                status_array = np.array([p.poll() for p in processes])
            for p in processes:
                p.kill()
        logging.info(f"Simulations for round {round_id} completed")
        # Initialise data loader for training
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Setting up dataloaders for round {round_id}"
        )
        train_data, val_data, trainer_dir = setup_dataloader(
            store, simulator, conf, round_id
        )

        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Setting up trainer for round {round_id}"
        )
        trainer = setup_trainer(trainer_dir, conf, round_id)

        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Initialising network for round {round_id}"
        )
        network = init_network(conf)

        if (
            not conf["tmnre"]["infer_only"]
            or len(glob.glob(f"{trainer_dir}/epoch*_R{round_id}.ckpt")) == 0
        ):
            print(
                f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Training network for round {round_id}"
            )
            trainer.fit(network, train_data, val_data)
            logging.info(
                f"Training completed for round {round_id}, checkpoint available at {glob.glob(f'{trainer_dir}/epoch*_R{round_id}.ckpt')[0]}"
            )

        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Generate prior samples"
        )
        prior_samples = simulator.sample(100_000, targets=["z"])

        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Generate posterior samples"
        )
        trainer.test(
            network, val_data, glob.glob(f"{trainer_dir}/epoch*_R{round_id}.ckpt")[0]
        )
        logratios = trainer.infer(
            network, obs, prior_samples.get_dataloader(batch_size=2048)
        )
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Saving logratios from round {round_id}"
        )
        save_logratios(logratios, conf, round_id)
        logging.info(f"Logratios saved for round {round_id}")

        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Update bounds from round {round_id}"
        )
        bounds_object = sl.bounds.get_rect_bounds(
            logratios, threshold=conf["tmnre"]["bounds_th"]
        )
        if type(bounds_object) == list:
            bounds = np.squeeze(bounds_object[0].bounds.numpy())
        else:
            bounds = np.squeeze(bounds_object.bounds.numpy())
        # Update bounds from bounds object
        update_bounds(bounds, conf, round_id)
        end_time = datetime.now()
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Completed round {round_id} in {end_time - start_time}."
        )
        logging.info(f"Completed round {round_id}")
    # Exit the program successfully
    sys.exit(0)


if __name__ == "__main__":
    args = sys.argv[1:]
    try:
        main(args)
    except KeyboardInterrupt:
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Program interrupted by user."
        )
        sys.exit(1)
