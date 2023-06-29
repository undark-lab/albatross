import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import swyft.lightning as sl
torch.set_float32_matmul_precision("high")
from swyft.lightning.estimators import LogRatioEstimator_Autoregressive


class InferenceNetwork(sl.SwyftModule):
    def __init__(self, conf):
        super().__init__()
        self.batch_size = conf["hparams"]["training_batch_size"]
        self.num_params = len(conf["vary_idxs"])
        self.vary_idxs = conf["vary_idxs"]
        self.one_d_only = conf["tmnre"]["1d_only"]
        self.network_options = conf["network"]
        self.in_channels = self.network_options["in_channels"]
        self.unet = Unet(in_channels=len(self.in_channels), out_channels=1)
        self.flatten = nn.Flatten(1)
        self.param_order = [8, 9, 10, 11, 12, 13, 14, 0, 15, 1, 2, 3, 4, 5, 6, 7]
        #self.param_order = [8, 9, 10, 11, 12, 13, 14, 0, 15, 1, 2, 3, 4, 5, 6, 7] v14
        self.n_features = len(self.param_order) * (len(self.param_order) - 1) 
        self.linear_1d = LinearCompression(self.n_features)
        self.lre = LogRatioEstimator_Autoregressive(
             self.n_features, len(self.param_order), "z"
        )
        self.noise_shuffling = conf["tmnre"]["shuffling"]
        self.optimizer_init = sl.AdamOptimizerInit(lr=conf["hparams"]["learning_rate"])

    def forward(self, A, B):
        if self.noise_shuffling and A["stream"].size(0) > 1:
            noise_shuffling = torch.randperm(self.batch_size)
            background = A["background"][noise_shuffling]
            img = A["stream"][:, self.in_channels] + background[:, self.in_channels]
        else:
            img = (
                A["stream"][:, self.in_channels] + A["background"][:, self.in_channels]
            )
        s_min, s_max = 0.0, 32.0
        img = torch.clamp((img - s_min) / (s_max - s_min), min=0.0, max=2.0)
        img = self.unet(img)
        compression = self.linear_1d(img)
        z_tot_A = A["z"][:, self.param_order]
        z_tot_B = B["z"][:, self.param_order]
        logratios = self.lre(compression, z_tot_A, z_tot_B)
        return logratios
    
# s_min, s_max = 0.0, 32.0
# img = torch.clamp((img - s_min) / (s_max - s_min), min=0.0, max=2.0)
# img = self.unet(img)
# compression = self.linear_1d(img)
# z_tot_A = A["z"][:, self.param_order]
# z_tot_B = B["z"][:, self.param_order]
# logratios = self.lre(compression, z_tot_A, z_tot_B)
# return logratios

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()
        self.inc = DoubleConv(in_channels, 4)
        self.down = Down(4, 8)
        self.up = Up(8, 4, False)
        self.outc = OutConv(4, out_channels)
        self.batch_norm = nn.LazyBatchNorm2d()

    def forward(self, x):
        x = self.batch_norm(x)
        x0 = self.inc(x)
        x1 = self.down(x0)
        up = self.up(x1, x0)
        f = self.outc(up)
        return f


class LinearCompression(nn.Module):
    def __init__(self, out_channels=16):
        super(LinearCompression, self).__init__()
        self.image_compression = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.LazyConv2d(out_channels=1, kernel_size=2),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=1, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.linear_compression = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.LazyLinear(out_channels),
        )

    def forward(self, x):
        img_compression = self.image_compression(x)
        return self.linear_compression(img_compression)


# class Unet(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Unet, self).__init__()
#         self.inc = DoubleConv(in_channels, 4)
#         self.down = Down(4, 8)
#         self.up = Up(8, 4, False)
#         self.outc = OutConv(4, out_channels)
#         self.batch_norm = nn.LazyBatchNorm2d()

#     def forward(self, x):
#         x = self.batch_norm(x)
#         x0 = self.inc(x)
#         x1 = self.down(x0)
#         up = self.up(x1, x0)
#         f = self.outc(up)
#         return f


# class LinearCompression(nn.Module):
#     def __init__(self, out_channels=16):
#         super(LinearCompression, self).__init__()
#         self.image_compression = nn.Sequential(
#             nn.LazyBatchNorm2d(),
#             nn.LazyConv2d(out_channels=1, kernel_size=2),
#             nn.ReLU(),
#             nn.LazyConv2d(out_channels=1, kernel_size=2),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#         )
#         self.linear_compression = nn.Sequential(
#             nn.Flatten(),
#             nn.LazyLinear(128),
#             nn.ReLU(),
#             nn.LazyLinear(64),
#             nn.ReLU(),
#             nn.LazyLinear(out_channels),
#         )

#     def forward(self, x):
#         img_compression = self.image_compression(x)
#         return self.linear_compression(img_compression)
        
        


def init_network(conf: dict):
    """
    Initialise the network with the settings given in a loaded config dictionary
    Args:
      conf: dictionary of config options, output of init_config
    Returns:
      Pytorch lightning network class
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> conf = init_config(tmnre_parser, sysargs, sim=True)
    >>> network = init_network(conf)
    """
    network = InferenceNetwork(conf)
    return network


def setup_zarr_store(conf, simulator, round_id=None, coverage=False, n_sims=None):
    """
    Initialise or load a zarr store for saving simulations
    Args:
      conf: dictionary of config options, output of init_config
      simulator: simulator object, output of init_simulator
      round_id: specific round id for store name
      coverage: specifies if store should be used for coverage sims
      n_sims: number of simulations to initialise store with
    Returns:
      Zarr store object
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> conf = init_config(tmnre_parser, sysargs, sim=True)
    >>> simulator = init_simulator(conf)
    >>> store = setup_zarr_store(conf, simulator)
    """
    zarr_params = conf["zarr_params"]
    if zarr_params["use_zarr"]:
        chunk_size = zarr_params["chunk_size"]
        if n_sims is None:
            if "nsims" in zarr_params.keys():
                n_sims = zarr_params["nsims"]
            else:
                n_sims = zarr_params["sim_schedule"][round_id - 1]
        shapes, dtypes = simulator.get_shapes_and_dtypes()
        store_path = zarr_params["store_path"]
        if round_id is not None:
            if coverage:
                store_dir = f"{store_path}/coverage_simulations_{zarr_params['run_id']}_R{round_id}"
            else:
                store_dir = (
                    f"{store_path}/simulations_{zarr_params['run_id']}_R{round_id}"
                )
        else:
            if coverage:
                store_dir = (
                    f"{store_path}/coverage_simulations_{zarr_params['run_id']}_R1"
                )
            else:
                store_dir = f"{store_path}/simulations_{zarr_params['run_id']}_R1"

        store = sl.ZarrStore(f"{store_dir}")
        store.init(N=n_sims, chunk_size=chunk_size, shapes=shapes, dtypes=dtypes)
        return store
    else:
        return None


def setup_dataloader(store, simulator, conf, round_id=None):
    """
    Initialise a dataloader to read in simulations from a zarr store
    Args:
      store: zarr store to load from, output of setup_zarr_store
      conf: dictionary of config options, output of init_config
      simulator: simulator object, output of init_simulator
      round_id: specific round id for store name
    Returns:
      (training dataloader, validation dataloader), trainer directory
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> conf = init_config(tmnre_parser, sysargs, sim=True)
    >>> simulator = init_simulator(conf)
    >>> store = setup_zarr_store(conf, simulator)
    >>> train
    """
    if round_id is not None:
        trainer_dir = f"{conf['zarr_params']['store_path']}/trainer_{conf['zarr_params']['run_id']}_R{round_id}"
    else:
        trainer_dir = f"{conf['zarr_params']['store_path']}/trainer_{conf['zarr_params']['run_id']}_R1"
    if not os.path.isdir(trainer_dir):
        os.mkdir(trainer_dir)
    hparams = conf["hparams"]
    if conf["tmnre"]["resampler"]:
        resampler = simulator.get_resampler(targets=conf["tmnre"]["noise_targets"])
    else:
        resampler = None
    train_data = store.get_dataloader(
        num_workers=int(hparams["num_workers"]),
        batch_size=int(hparams["training_batch_size"]),
        idx_range=[0, int(hparams["train_data"] * len(store.data.z))],
        on_after_load_sample=resampler,
    )
    val_data = store.get_dataloader(
        num_workers=int(hparams["num_workers"]),
        batch_size=int(hparams["validation_batch_size"]),
        idx_range=[
            int(hparams["train_data"] * len(store.data.z)),
            len(store.data.z) - 1,
        ],
        on_after_load_sample=None,
    )
    return train_data, val_data, trainer_dir


def setup_trainer(trainer_dir, conf, round_id):
    """
    Initialise a pytorch lightning trainer and relevant directories
    Args:
      trainer_dir: location for the training logs
      conf: dictionary of config options, output of init_config
      round_id: specific round id for store name
    Returns:
      Swyft lightning trainer instance
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> conf = init_config(tmnre_parser, sysargs, sim=True)
    >>> simulator = init_simulator(conf)
    >>> store = setup_zarr_store(conf, simulator, 1)
    >>> train_data, val_data, trainer_dir = setup_dataloader(store, simulator, conf, 1)
    >>> trainer = setup_trainer(trainer_dir, conf, 1)
    """
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.0, patience=7, verbose=False, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{trainer_dir}",
        filename="{epoch}_{val_loss:.2f}_{train_loss:.2f}" + f"_R{round_id}",
        mode="min",
    )
    logger_tbl = pl_loggers.TensorBoardLogger(
        save_dir=f"{trainer_dir}",
        name=f"{conf['zarr_params']['run_id']}_R{round_id}",
        version=None,
        default_hp_metric=False,
    )

    device_params = conf["device_params"]
    hparams = conf["hparams"]
    trainer = sl.SwyftTrainer(
        accelerator=device_params["device"],
        gpus=device_params["n_devices"],
        min_epochs=hparams["min_epochs"],
        max_epochs=hparams["max_epochs"],
        logger=logger_tbl,
        callbacks=[lr_monitor, early_stopping_callback, checkpoint_callback],
    )
    return trainer


def save_logratios(logratios, conf, round_id):
    """
    Save logratios from a particular round
    Args:
      logratios: swyft logratios instance
      conf: dictionary of config options, output of init_config
      round_id: specific round id for store name
    """
    if not os.path.isdir(
        f"{conf['zarr_params']['store_path']}/logratios_{conf['zarr_params']['run_id']}"
    ):
        os.mkdir(
            f"{conf['zarr_params']['store_path']}/logratios_{conf['zarr_params']['run_id']}"
        )
    with open(
        f"{conf['zarr_params']['store_path']}/logratios_{conf['zarr_params']['run_id']}/logratios_R{round_id}",
        "wb",
    ) as p:
        pickle.dump(logratios, p)


def save_coverage(coverage, conf, round_id):
    """
    Save bounds from a particular round
    Args:
      bounds: unpacked swyft bounds object
      conf: dictionary of config options, output of init_config
      round_id: specific round id for store name
    """
    if not os.path.isdir(
        f"{conf['zarr_params']['store_path']}/coverage_{conf['zarr_params']['run_id']}"
    ):
        os.mkdir(
            f"{conf['zarr_params']['store_path']}/coverage_{conf['zarr_params']['run_id']}"
        )
    with open(
        f"{conf['zarr_params']['store_path']}/coverage_{conf['zarr_params']['run_id']}/coverage_R{round_id}",
        "wb",
    ) as p:
        pickle.dump(coverage, p)


def update_bounds(bounds, conf, round_id):
    """
    Load bounds from a particular round
    Args:
      conf: dictionary of config options, output of init_config
      round_id: specific round id for store name
    Returns:
      Bounds object with ordering defined by the param idxs in the config
    """
    np.savetxt(
        f"{conf['zarr_params']['store_path']}/bounds_{conf['zarr_params']['run_id']}_R{round_id}.txt",
        bounds,
    )
    for idx, param_id in enumerate(conf["vary_idxs"]):
        key = conf["param_names"][param_id]
        if key in conf["varying"]:
            if key in ["nhalos"]:
                conf["priors"][key] = [int(bounds[idx, 0]), int(bounds[idx, 1])]
            else:
                conf["priors"][key] = [float(bounds[idx, 0]), float(bounds[idx, 1])]


def update_sim_bounds(conf, round_id):
    """
    Save coverage samples from a particular round
    Args:
      coverage: swyft coverage object instance
      conf: dictionary of config options, output of init_config
      round_id: specific round id for store name
    """
    if round_id != 1:
        bounds = np.loadtxt(
            f"{conf['zarr_params']['store_path']}/bounds_{conf['zarr_params']['run_id']}_R{round_id - 1}.txt"
        )
        for idx, param_id in enumerate(conf["vary_idxs"]):
            key = conf["param_names"][param_id]
            if key in conf["varying"]:
                if key in ["nhalos"]:
                    conf["priors"][key] = [int(bounds[idx, 0]), int(bounds[idx, 1])]
                else:
                    conf["priors"][key] = [float(bounds[idx, 0]), float(bounds[idx, 1])]


# Unet implementation follows below


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)