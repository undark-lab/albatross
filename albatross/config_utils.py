import configparser
import subprocess
from distutils.util import strtobool
from pathlib import Path
from datetime import datetime


def read_config(sysargs: list):
    """
    Load the config using configparser, returns a parser object that can be accessed as
    e.g. tmnre_parser['FIELD']['parameter'], this will always return a string, so must be
    parsed for data types separately, see init_config
    Args:
      sysargs: list of command line arguments (i.e. strings) containing path to config in position 0
    Returns:
      Config parser object containing information in configuration file sections
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> tmnre_parser['TMNRE']['num_rounds'] etc.
    """
    tmnre_config_file = sysargs[0]
    tmnre_parser = configparser.ConfigParser()
    tmnre_parser.read_file(open(tmnre_config_file))
    return tmnre_parser


def init_config(tmnre_parser, sysargs: list, sim: bool = False) -> dict:
    """
    Initialise the config dictionary, this is a dictionary of dictionaries obtaining by parsing
    the relevant config file. A copy of the config file is stored along with the eventual simulations.
    All parameters are parsed to the correct data type from strings, including lists and booleans etc.
    Args:
      tmnre_parser: config parser object, output of read_config
      sysargs: list of command line arguments (i.e. strings) containing path to config in position 0
      sim: boolean to choose whether to include config copying features etc. if False, will create a
           copy of the config in the store directory and generate the param idxs file
    Returns:
      Dictionary of configuration options with all data types explicitly parsed
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> conf = init_config(tmnre_parser, sysargs, sim=True)
    """
    tmnre_config_file = sysargs[0]
    store_path = tmnre_parser["ZARR PARAMS"]["store_path"]
    if not sim:
        Path(store_path).mkdir(parents=True, exist_ok=True)
    run_id = tmnre_parser["ZARR PARAMS"]["run_id"]
    if not sim:
        subprocess.run(
            f"cp {tmnre_config_file} {store_path}/config_{run_id}.txt", shell=True
        )
    conf = {}

    priors = {}
    true_values = {}
    fixed = []
    varying = []
    for key in tmnre_parser["PRIORS"]:
        if len(tmnre_parser["PRIORS"][key].split(",")) == 1:
            low = float(tmnre_parser["PRIORS"][key])
            high = float(tmnre_parser["PRIORS"][key])
            fixed.append(key)
            priors[key] = [low, high]
        else:
            low = float(tmnre_parser["PRIORS"][key].split(",")[0])
            high = float(tmnre_parser["PRIORS"][key].split(",")[1])
            varying.append(key)
            priors[key] = [low, high]
        true_values[key] = float(tmnre_parser["TRUE VALUES"][key])
    conf["priors"] = priors
    conf["fixed"] = fixed
    conf["varying"] = varying
    conf["true_values"] = true_values

    param_idxs = {}
    param_names = {}
    if not sim:
        with open(
            (
                f"{tmnre_parser['ZARR PARAMS']['store_path']}/param_idxs_{tmnre_parser['ZARR PARAMS']['run_id']}.txt"
            ),
            "w",
        ) as f:
            for idx, key in enumerate(priors.keys()):
                param_idxs[key] = idx
                param_names[idx] = key
                if key in varying:
                    f.write(f"{key} {idx} varying\n")
                else:
                    f.write(f"{key} {idx} fixed\n")
            f.close()
    else:
        for idx, key in enumerate(priors.keys()):
            param_idxs[key] = idx
            param_names[idx] = key
    conf["param_idxs"] = param_idxs
    conf["vary_idxs"] = [param_idxs[key] for key in varying]
    conf["param_names"] = param_names

    binning = {}
    for key in tmnre_parser["BINNING"]:
        if key in ["phi1", "phi2", "pm_phi1_cosphi2", "pm_phi2", "vrad", "dist"]:
            binning[key] = [
                float(val) for val in tmnre_parser["BINNING"][key].split(",")
            ]
        elif key in ["nbins"]:
            binning[key] = [int(val) for val in tmnre_parser["BINNING"][key].split(",")]
    conf["binning"] = binning

    errors = {}
    for key in tmnre_parser["ERRORS"]:
        if key in [
            "phi1",
            "phi2",
            "pm_phi1_cosphi2",
            "pm_phi2",
            "vrad",
            "dist",
            "stream_selection",
            "total_background",
            "background_removal",
        ]:
            errors[key] = float(tmnre_parser["ERRORS"][key])
    conf["errors"] = errors

    tmnre = {}
    tmnre["infer_only"] = False
    tmnre["marginals"] = None
    for key in tmnre_parser["TMNRE"]:
        if key in ["num_rounds"]:
            tmnre[key] = int(tmnre_parser["TMNRE"][key])
        elif key in ["bounds_th"]:
            tmnre[key] = float(tmnre_parser["TMNRE"][key])
        elif key in ["1d_only", "generate_obs", "resampler", "infer_only", "shuffling"]:
            tmnre[key] = bool(strtobool(tmnre_parser["TMNRE"][key]))
        elif key in ["obs_path"]:
            tmnre[key] = str(tmnre_parser["TMNRE"][key])
        elif key in ["noise_targets"]:
            tmnre[key] = [
                str(target) for target in tmnre_parser["TMNRE"][key].split(",")
            ]
        elif key in ["marginals"]:
            marginals_string = tmnre_parser["TMNRE"][key]
            marginals_list = []
            for marginal in marginals_string.split("("):
                split_marginal = marginal.split(")")
                if split_marginal[0] != "":
                    indices = split_marginal[0].split(",")
                    mg = []
                    for index in indices:
                        mg.append(int(index.strip(" ")))
                    marginals_list.append(tuple(mg))
            tmnre["marginals"] = tuple(marginals_list)
    conf["tmnre"] = tmnre

    zarr_params = {}
    for key in tmnre_parser["ZARR PARAMS"]:
        if key in ["run_id", "store_path"]:
            zarr_params[key] = tmnre_parser["ZARR PARAMS"][key]
        elif key in ["use_zarr", "run_parallel"]:
            zarr_params[key] = bool(strtobool(tmnre_parser["ZARR PARAMS"][key]))
        elif key in ["nsims", "chunk_size", "njobs"]:
            zarr_params[key] = int(tmnre_parser["ZARR PARAMS"][key])
        elif key in ["targets"]:
            zarr_params[key] = [
                target
                for target in tmnre_parser["ZARR PARAMS"][key].split(",")
                if target != ""
            ]
        elif key in ["sim_schedule"]:
            zarr_params[key] = [
                int(nsims) for nsims in tmnre_parser["ZARR PARAMS"][key].split(",")
            ]
    if "sim_schedule" in zarr_params.keys():
        if len(zarr_params["sim_schedule"]) != tmnre["num_rounds"]:
            print(
                f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [config_utils.py] | WARNING: Error in sim scheduler, setting to default n_sims = 30_000"
            )
            zarr_params["nsims"] = 3_000
        elif "nsims" in zarr_params.keys():
            zarr_params.pop("nsims")
    conf["zarr_params"] = zarr_params

    network = {}
    for key in tmnre_parser["NETWORK"].keys():
        if key in ["in_channels"]:
            channels = [
                str(channel) for channel in tmnre_parser["NETWORK"][key].split(",")
            ]
            in_channels = []
            for channel in channels:
                if channel == "pos":
                    in_channels.append(0)
                elif channel == "vel":
                    in_channels.append(1)
                elif channel == "rad":
                    in_channels.append(2)
            network["in_channels"] = in_channels
        elif key in ["num_1d_features", "num_2d_features"]:
            network[key] = int(tmnre_parser["NETWORK"][key])
    conf["network"] = network

    hparams = {}
    for key in tmnre_parser["HYPERPARAMS"].keys():
        if key in [
            "min_epochs",
            "max_epochs",
            "early_stopping",
            "num_workers",
            "training_batch_size",
            "validation_batch_size",
        ]:
            hparams[key] = int(tmnre_parser["HYPERPARAMS"][key])
        elif key in ["learning_rate", "train_data", "val_data"]:
            hparams[key] = float(tmnre_parser["HYPERPARAMS"][key])
    conf["hparams"] = hparams

    device_params = {}
    for key in tmnre_parser["DEVICE PARAMS"]:
        if key in ["device"]:
            device_params[key] = str(tmnre_parser["DEVICE PARAMS"][key])
        elif key in ["n_devices"]:
            device_params[key] = int(tmnre_parser["DEVICE PARAMS"][key])
    conf["device_params"] = device_params

    return conf
