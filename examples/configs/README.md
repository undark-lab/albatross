# Setting up an *albatross* configuration file

### **Injection parameters:**
```
[TRUE VALUES]
age = 3000.
lrelease = 1.405
lmatch = 1.846
xi0 = 0.001
alpha = 20.9
rh = 0.001
mbar = 3.
logmsat = 4.05
sigv = 1.1
xc = 11.8
yc = 0.79
zc = 6.4
vxc = 109.5
vyc = -254.5
vzc = -90.3
stripnear = 0.5
```

- All entries follow the `par_name = par_value` structure
- More complicated structures (e.g. lists of parameters) can be accomodated by modifying the corresponding part of the config reader in `config_utils.py`
- If you want to generate the observation, these parameters will be your ground truth
- If you want to analyse a pre-existing stellar stream observation, set `generate_obs` in `[TMNRE]` to `False` (In this case, the injection values are irrelevant). Make sure to change `obs_path` to point to your observation of interest.
- Full descriptions of the parameters can be found in the `sstrax` repository in e.g. the [`Parameters` dataclass](https://github.com/undark-lab/sstrax/blob/55d3c31acc6fbe07fd204788b616f67bdd161b96/sstrax/constants.py#L157) or in the accompanying paper.

### **Prior limits**
```
[PRIORS]
age = 500.,5000.
lrelease = 0.1,2.0
lmatch = 0.1,2.0
xi0 = 0.0001, 0.01
alpha = 10.,30.
rh = 0.0001,0.01
mbar = 1.,20.
logmsat = 3.,4.5
sigv = 0.1,5.
xc = 10.,14.
yc = 0.1,2.5
zc = 6.,8.
vxc = 90.,115.
vyc = -280.,-230.
vzc = -120.,-80.
stripnear = 0.0,1.0
```
- Note that the order of these parameters reflect the ordering in your final results, which will be specified in the `param_idxs_[run_id].txt` file that is generated automatically.
- **Varying parameters:** Follows `par_name = lower_bound,upper_bound` format
- **Fixed parameters:** Follows `par_name = par_value` format. Be careful to match the injected value if the observation is generated from the true values above.
- Full descriptions of the parameters can be found in the `sstrax` repository in e.g. the [`Parameters` dataclass](https://github.com/undark-lab/sstrax/blob/55d3c31acc6fbe07fd204788b616f67bdd161b96/sstrax/constants.py#L157) or in the accompanying paper.
- **Please do not leave spaces after the commas to avoid parsing errors**

### **Binning Options**
```
[BINNING]
phi1 = -120.,70.
phi2 = -8.,2.
pm_phi1_cosphi2 = -2.,1.
pm_phi2 = -0.1,0.1
vrad = -250.,250.
dist = 6.,20.
nbins = 64,32
```
- `phi1` | Type: `float`, `float` | Observation window in the $\phi_1$ co-ordinate (meaured in $\mathrm{deg}$)
- `phi2` | Type: `float`, `float` | Observation window in the $\phi_2$ co-ordinate (meaured in $\mathrm{deg}$)
- `pm_phi1_cosphi2` | Type: `float`, `float` | Observation window in the $\mu_{\phi_1}\cos\phi_2$ co-ordinate (meaured in $\mathrm{mas/yr}$)
- `pm_phi2` | Type: `float`, `float` | Observation window in the $\mu_{\phi_2}$ co-ordinate (meaured in $\mathrm{mas/yr}$)
- `vrad` | Type: `float`, `float` | Observation window in the $v_\mathrm{rad}$ co-ordinate (meaured in $\mathrm{km/s}$)
- `dist` | Type: `float`, `float` | Observation window in the $d$ co-ordinate (meaured in $\mathrm{kpc}$)
- `nbins` | Type: `int`, `int` | Number of horizontal and vertical bins for each image plane ($\phi_1, \phi_2$), ($\mu_{\phi_1}\cos\phi_2, \mu_{\phi_2}$) and ($v_\mathrm{rad}, d$)

### **Observational Errors**
```
[ERRORS]
phi1 = 0.001
phi2 = 0.15
pm_phi1_cosphi2 = 0.1
pm_phi2 = 0.0
vrad = 5.
dist = 0.25
stream_selection = 0.95
total_background = 2e6
background_removal = 0.00001
```
- `phi1` | Type: `float` | Observation error in the $\phi_1$ co-ordinate (meaured in $\mathrm{deg}$)
- `phi2` | Type: `float` | Observation error in the $\phi_2$ co-ordinate (meaured in $\mathrm{deg}$)
- `pm_phi1_cosphi2` | Type: `float` | Observation error in the $\mu_{\phi_1}\cos\phi_2$ co-ordinate (meaured in $\mathrm{mas/yr}$)
- `pm_phi2` | Type: `float` | Observation error in the $\mu_{\phi_2}$ co-ordinate (meaured in $\mathrm{mas/yr}$)
- `vrad` | Type: `float` | Observation error in the $v_\mathrm{rad}$ co-ordinate (meaured in $\mathrm{km/s}$)
- `dist` | Type: `float` | Observation error in the $d$ co-ordinate (meaured in $\mathrm{kpc}$)
- `stream_selection` | Type: `float` in [0, 1] | Input selection efficiency for subsampling the generated stream stars (randomly selects this percentage of the set of individual stars to model some selection model)
- `total_background` | Type: `int` | Total number of background stars in the observing window
- `background_removal` | Type: `float` | Fraction of background stars incorrectly included in stream as a result of some selection model


### **Parameters defining the (`zarr`) store for the stream simulations**
```
[ZARR PARAMS]
run_id = albatross_example
use_zarr = True
sim_schedule = 30_000,30_000,30_000,30_000,30_000,60_000,150_000
chunk_size = 10
run_parallel = True
njobs = 128
targets = z,stream,background
store_path = /path/to/data/store/albatross_example
run_description = Example config for running albatross v0.0.1
```
- `run_id` | Type: `str` | Unique identifier for the **albatross** run (names the output directory and result files)
- `use_zarr` | Type: `bool` | Option to use a zarr store for storing simulations (recommended setting: `True`)
- `sim_schedule` | Type: `int` | Schedule for number of simulations per round of **albatross**-tmnre 
    - Follows : `n_sims_R1`,`n_sims_R2`,...,`n_sims_RN` where `N` is the number of rounds
- `chunk_size` | Type: `int` | Number of simulations to generate per batch
- `run_parallel` | Type: `bool` | Option to simulate parallely across cpus
- `njobs` | Type: `int` | number of parallel simulation threads (Defaults to `n_cpus` if `njobs` > `n_cpus` of your machine)
- `targets` | Type: `str` | Targets to be simulated by the **sstrax** simulator
    - `z`: Parameter samples from the prior
    - `stream`: Binned image generated from the `sstrax` simulator
    - `background`: Binned image of the background stars following the observational model choices
- `store_path` | Type: `str` | Path to the directory to store the simulations
- `run_description` | Type: `str` | Description for the specific **albatross** run
- **Please do not leave spaces after the commas to avoid parsing errors**

### **TMNRE parameters**
```
[TMNRE]
num_rounds = 7
1d_only = True
infer_only = True
marginals = (0, 1)
bounds_th = 1e-5
resampler = False
shuffling = True
noise_targets = background
generate_obs = False
obs_path = /path/to/observation/observation_albatross_example
```
- `num_rounds` | Type: `int` | Number of TMNRE rounds to be executed
- `1d_only` | Type: `bool` | Choice of training only the 1D marginals (if `True`, neglects the `marginals` argument)
- `infer_only` | Type: `bool` | Choice for running only inference if you have a pretrained NN
- `marginals` | Type: `tuple` | If `1d_only` is set to `False`, specify the higher dimensional marginals that you want to train
- `bounds_th` | Type: `float` | Threshold determining the bounds for each round of truncation. ($\epsilon$ defined in [Cole et al.](https://arxiv.org/abs/2111.08030))
- `resample` | Type: `bool` | Choice for resampling the noise realizations at each training iteration (slow!)
- `shuffling` | Type: `bool` | Choice of shuffling the noise realizations within taining batches (fast!). Same purpose as the noise resampler but faster if you have the noise strains sampled along with the simulations (See `targets` in `[ZARR PARAMS]`) 
- `noise_targets` | Type: `str` | Noise targets to be used (Should comply with the data strains used for training (Default: `background`)
- `generate_obs` | Type: `bool` | Choice to generate the observation before training
- `obs_path` | Type: `str` | Path to observation file (loaded as a pickle object) when `generate_obs` is `False`. Default for a generated observtion is `store_path/observation_run_id`

### **Network choices**
```
[NETWORK]
in_channels = pos,vel,rad
num_1d_features = 32
num_2d_features = 128
```
- `in_channels` | Type: `list(str)` | Options: `pos`, `vel`, `rad` | Choice of input channels to show the network, can be any combination of these options separated by commas (but no spaces to avoid parsing errors)
- `num_1d_features` | Type: `int` | Number of features to show to the `swyft` logratio estimator when estimating 1d marginals
- `num_2d_features` | Type: `int` | Number of features to show to the `swyft` logratio estimator when estimating higher dimensional marginals

### **Hyperparameters for training the NN**
```
[HYPERPARAMS]
min_epochs = 0
max_epochs = 50
early_stopping = 20
learning_rate = 5e-4
num_workers = 6
training_batch_size = 64
validation_batch_size = 64
train_data = 0.9
val_data = 0.1
```
- `min_epochs` | Type: `int` | Minimum number of epochs to train for
- `max_epochs` | Type: `int` | Maximum number of epochs to train for
- `early_stopping` | Type: `int` | Number of training epochs to wait before stopping training in case of overfitting (reverts to the last minimum validation loss epoch)
- `learning_rate` | Type: `float` | The initial learning rate of the trainer
- `num_workers` | Type: `int` | Number of worker processes for loading training and validation data
- `training_bath_size` | Type: `int` | Batch size of the training data to be passed on to the dataloader
- `validation_bath_size` | Type: `int` | Batch size of the validation data to be passed on to the dataloader
- `train_data` | Type: `float` | $\in$ [0,1], fraction of simulation data to be used for training
- `val_data` | Type: `float` | $\in$ [0,1], fraction of simulation data to be used for validation/testing. `train_data + val_data` must be less than 1.

### **Device parameters for training the NN**
```
[DEVICE PARAMS]
device = gpu
n_devices = 1
```
- `device` | Type: `str` | Device on which training is executed (Choice between `gpu` or `cpu`)
- `n_devices` | Type: `int` | Number of devices that the training can be parallelized over
---
