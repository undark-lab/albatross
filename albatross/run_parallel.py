import sstrax as st
from config_utils import read_config, init_config
from simulator_utils import init_simulator, simulate
from inference_utils import setup_zarr_store, update_sim_bounds
import sys
import psutil
from datetime import datetime


def main(round_id, args):
    if "coverage" not in args:
        coverage = False
    else:
        coverage = True
    tmnre_parser = read_config(args)
    conf = init_config(tmnre_parser, args, sim=True)
    update_sim_bounds(conf, round_id)
    simulator = init_simulator(
        st.simulate_stream, st.halo_to_gd1_vmap, st.halo_to_gd1_velocity_vmap, conf
    )
    store = setup_zarr_store(conf, simulator, round_id=round_id, coverage=coverage)
    while psutil.virtual_memory().percent < 80.0 and store.sims_required > 0:
        simulate(
            simulator, store, conf, max_sims=int(conf["zarr_params"]["chunk_size"])
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    round_id = int(args[1])
    try:
        main(round_id, args)
    except KeyboardInterrupt:
        print(
            f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [run_parallel.py] | Batch process terminated by keyboard interrupt."
        )
        sys.exit(1)
