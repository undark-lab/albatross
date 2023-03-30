import numpy as np
import swyft.lightning as sl
import jax
import random
from sstrax import Parameters, constants


class Simulator(sl.Simulator):
    def __init__(
        self,
        stream_generator,
        gd1_pos_converter,
        gd1_velocity_converter,
        priors,
        binning,
        errors,
        true_values,
    ):
        super().__init__()
        self.stream_generator = stream_generator
        self.gd1_pos_converter = gd1_pos_converter
        self.gd1_velocity_converter = gd1_velocity_converter
        self.priors = priors
        self.binning = binning
        self.errors = errors
        self.true_values = true_values
        self.transform_samples = sl.to_numpy32

    def build(self, graph):
        """
        Define the computational graph, which allows us to sample specific targets efficiently
        """
        z = graph.node("z", self.sample_prior)
        stream = graph.node("stream", self.generate_stream, z)
        background = graph.node("background", self.generate_background)

    def generate_observation(self):
        """
        Generate the observations from the true values of the parameters
        """
        z = []
        true_values = self.true_values.copy()
        for key in true_values.keys():
            if key in ["nhalos"]:
                z.append(int(true_values[key]))
            if key in ["logmsat"]:
                z.append(true_values[key])
            else:
                z.append(true_values[key])
        return self.sample(conditions={"z": np.array(z)})

    def sample_prior(self):
        """
        Specific function to sample from the priors defined in the config file.
        """
        sample_dict = {}
        for key in self.priors.keys():
            sample_dict[key] = np.random.uniform(
                self.priors[key][0], self.priors[key][1], 1
            )[0]
        return np.array(list(sample_dict.values()))

    def run_stream_generator(self, z):
        """
        Has to be built separately than in the graph scope because the julia function takes a
        dictionary of parameters as input, rather than an array that is generated in the graph
        """
        ps = self.true_values.copy()
        for idx, key in enumerate(self.priors.keys()):
            if key in ["nhalos"]:
                ps[key] = int(z[idx])
            if key in ["logmsat"]:
                ps["msat"] = np.power(10.0, z[idx])
            else:
                ps[key] = z[idx]
        theta = Parameters(**{k: ps[k] for k in constants.PRIOR_LIST})
        _rkey = jax.random.PRNGKey(random.randint(0, 1e8))
        stream = self.stream_generator(_rkey, theta)
        return stream

    def stars_to_gd1(self, stars):
        """
        Convert the simulated stars to GD1 coordinates
        """
        Xhalo = stars[:, :3]
        Vhalo = stars[:, 3:]
        Xgd1 = np.array(
            self.gd1_pos_converter(Xhalo)
        )  # positions in kpc, time in Myr, velocities in kpc/Myr
        Vgd1 = np.array(self.gd1_velocity_converter(Xhalo, Vhalo))
        Vgd1[:, 0] = Vgd1[:, 0] * 977.7922216807891  # convert from kpc/Myr to km/s
        Vgd1[:, 1] = (
            Vgd1[:, 1] / Xgd1[:, 0] * 2.0626480624709636e8 / 1e6
        )  # pm_phi1_cosphi2: converted from rad/Myr to mas/yr
        Vgd1[:, 2] = (
            Vgd1[:, 2] / Xgd1[:, 0] * 2.0626480624709636e8 / 1e6
        )  # pm_phi2: converted from rad/Myr to mas/yr
        Xgd1[:, 1] = Xgd1[:, 1] * 180.0 / np.pi  # convert from rad to deg
        Xgd1[:, 2] = Xgd1[:, 2] * 180.0 / np.pi  # convert from rad to deg
        return np.concatenate((Xgd1, Vgd1), axis=1)

    def add_noise(self, gd1):
        gd1[:, 0] = gd1[:, 0] + np.random.normal(
            0.0, self.errors["dist"], len(gd1[:, 0])
        )
        gd1[:, 1] = gd1[:, 1] + np.random.normal(
            0.0, self.errors["phi1"], len(gd1[:, 1])
        )
        gd1[:, 2] = gd1[:, 2] + np.random.normal(
            0.0, self.errors["phi2"], len(gd1[:, 2])
        )
        gd1[:, 3] = gd1[:, 3] + np.random.normal(
            0.0, self.errors["vrad"], len(gd1[:, 3])
        )
        gd1[:, 4] = gd1[:, 4] + np.random.normal(
            0.0, self.errors["pm_phi1_cosphi2"], len(gd1[:, 4])
        )
        gd1[:, 5] = gd1[:, 5] + np.random.normal(
            0.0, self.errors["pm_phi2"], len(gd1[:, 5])
        )
        stream_selection = gd1[
            np.random.choice(
                len(gd1[:, 0]),
                size=int(np.floor(self.errors["stream_selection"] * len(gd1[:, 0]))),
                replace=False,
            ),
            :,
        ]
        return stream_selection

    def bin(self, gd1):
        """
        Convert the simulated GD1 stars to binned images
        """
        dist = gd1[:, 0]
        phi1 = gd1[:, 1]
        phi2 = gd1[:, 2]
        vrad = gd1[:, 3]
        pm_phi1_cosphi2 = gd1[:, 4]
        pm_phi2 = gd1[:, 5]
        phi1_phi2, _, _ = np.histogram2d(
            phi1,
            phi2,
            bins=self.binning["nbins"],
            range=[self.binning["phi1"], self.binning["phi2"]],
            density=False,
        )
        pm_phi1_cosphi2_pm_phi2, _, _ = np.histogram2d(
            pm_phi1_cosphi2,
            pm_phi2,
            bins=self.binning["nbins"],
            range=[self.binning["pm_phi1_cosphi2"], self.binning["pm_phi2"]],
            density=False,
        )
        dist_vrad, _, _ = np.histogram2d(
            dist,
            vrad,
            bins=self.binning["nbins"],
            range=[self.binning["dist"], self.binning["vrad"]],
            density=False,
        )
        full_image = np.array([phi1_phi2, pm_phi1_cosphi2_pm_phi2, dist_vrad])
        return full_image

    def sample_background(self):
        num_stars = int(
            np.floor(
                self.errors["total_background"] * self.errors["background_removal"]
            )
        )
        gd1_background = np.zeros((num_stars, 6))
        gd1_background[:, 0] = np.random.uniform(
            self.binning["dist"][0], self.binning["dist"][1], num_stars
        )
        gd1_background[:, 1] = np.random.uniform(
            self.binning["phi1"][0], self.binning["phi1"][1], num_stars
        )
        gd1_background[:, 2] = np.random.uniform(
            self.binning["phi2"][0], self.binning["phi2"][1], num_stars
        )
        gd1_background[:, 3] = np.random.uniform(
            self.binning["vrad"][0], self.binning["vrad"][1], num_stars
        )
        gd1_background[:, 4] = np.random.uniform(
            self.binning["pm_phi1_cosphi2"][0],
            self.binning["pm_phi1_cosphi2"][1],
            num_stars,
        )
        gd1_background[:, 5] = np.random.uniform(
            self.binning["pm_phi2"][0], self.binning["pm_phi2"][1], num_stars
        )
        return gd1_background

    def generate_background(self):
        background_list = self.sample_background()
        return self.bin(background_list)

    def generate_stream(self, z):
        stars = self.run_stream_generator(z)
        stars_gd1 = self.stars_to_gd1(stars)
        stars_gd1 = self.add_noise(stars_gd1)
        return self.bin(stars_gd1)


def init_simulator(stream_generator, gd1_pos_converter, gd1_velocity_converter, conf):
    """
    Initialise the swyft simulator
    Args:
      stream_generator: function (from sstrax or elsewhere) to generate the stream
      gd1_pos_coverter: function (from sstrax or elsewhere) to convert to GD1 position
      gd1_vel_converter: function (from sstrax or elsewhere) to convert to GD1 velocity
      conf: dictionary of config options, output of init_config
    Returns:
      Swyft lightning simulator instance
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> conf = init_config(tmnre_parser, sysargs, sim=True)
    >>> import sstrax as st
    >>> simulator = init_simulator(st.simulate_stream, st.halo_to_gd1_vmap, st.halo_to_gd1_velocity_vmap, conf)
    """
    simulator = Simulator(
        stream_generator,
        gd1_pos_converter,
        gd1_velocity_converter,
        conf["priors"],
        conf["binning"],
        conf["errors"],
        conf["true_values"],
    )
    return simulator


def simulate(simulator, store, conf, max_sims=None):
    """
    Run a swyft simulator to save simulations into a given zarr store
    Args:
      simulator: swyft simulator object
      store: initialised zarr store
      conf: dictionary of config options, output of init_config
      max_sims: maximum number of simulations to perform (otherwise will fill store)
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> conf = init_config(tmnre_parser, sysargs, sim=True)
    >>> simulator = init_simulator(conf)
    >>> store = setup_zarr_store(conf, simulator)
    >>> simulate(simulator, store, conf)
    """
    store.simulate(
        sampler=simulator,
        batch_size=int(conf["zarr_params"]["chunk_size"]),
        max_sims=max_sims,
    )
