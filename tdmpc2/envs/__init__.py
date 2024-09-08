from copy import deepcopy
import warnings

import gym

from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.pixels import PixelWrapper
from envs.wrappers.tensor import TensorWrapper

def missing_dependencies(task):
	raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')

try:
	from .env_builder import make_env as make_humanoid_robot_env
except:
	make_locomotion_env = missing_dependencies

from .env_builder import _test
_test()

warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_multitask_env(cfg):
	"""
	Make a multi-task environment for TD-MPC2 experiments.
	"""
	print('Creating multi-task environment with tasks:', cfg.tasks)
	envs = []
	for task in cfg.tasks:
		_cfg = deepcopy(cfg)
		_cfg.task = task
		_cfg.multitask = False
		env = make_env(_cfg)
		if env is None:
			raise ValueError('Unknown task:', task)
		envs.append(env)
	env = MultitaskWrapper(cfg, envs)
	cfg.obs_shapes = env._obs_dims
	cfg.action_dims = env._action_dims
	cfg.episode_lengths = env._episode_lengths
	return env
	
def make_env(cfg):
	"""
	Make an environment for TD-MPC2 experiments.
	"""
	gym.logger.set_level(40)
	if cfg.multitask:
		env = make_multitask_env(cfg)
	else:
		env = None
		try:
			env = make_humanoid_robot_env(cfg)
		except ValueError:
			raise ValueError(f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.')
		env = TensorWrapper(env)
	if cfg.get('obs', 'state') == 'rgb':
		env = PixelWrapper(cfg, env)
	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.get_observation_space_agent().spaces.items()}
	except: # Box
		cfg.obs_shape = {cfg.get('obs', 'state'): env.get_observation_space_agent().shape}
	print("cfg.obs_shape",cfg.obs_shape)
	cfg.action_dim = env.get_action_space_shape_agent() #.shape[0]
	print("cfg.action_dim",cfg.action_dim)
	cfg.episode_length = env.max_episode_steps
	cfg.seed_steps = max(1000, 5*cfg.episode_length)
	return env
