from onpolicy.envs.starcraft2.StarCraft2_UnifiedEnv import StarCraft2UnifiedEnv as SC2Env
from onpolicy.debug_config import debug_get_config
from train_smac import parse_args
import numpy as np


parser = debug_get_config()
all_args = parse_args(None, parser)
all_args.map_name = "2c_vs_64zg"
env = SC2Env(all_args)
env.seed(0)
env.reset()


print(np.shape([env.get_obs_agent(i) for i in range(env.n_agents)]))

env.close()