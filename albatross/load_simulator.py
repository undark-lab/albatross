print(
    r"""
        A       Initialising albatross
    ___/_\___   ----------------------
     ',. ..'    Type: Load Simulator
     /.'^'.\    Authors: J. Alvey, M. Gerdes
    /'     '\   Version: v0.0.1 | April 2023
"""
)

import sys
from datetime import datetime
import sstrax as st
from config_utils import read_config, init_config
from simulator_utils import init_simulator

if __name__ == "__main__":
    args = sys.argv[1:]
    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [load_simulator.py] | Reading config file"
    )
    tmnre_parser = read_config(args)
    conf = init_config(tmnre_parser, args, sim=True)
    simulator = init_simulator(
        st.simulate_stream, st.halo_to_gd1_vmap, st.halo_to_gd1_velocity_vmap, conf
    )
