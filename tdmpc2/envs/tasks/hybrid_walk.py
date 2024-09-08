
from .task import Task
from ..robots.robot import Robot
from ..environment import Environment
from ..rewards import Reward
from .walk import Walk

import numpy as np
from dm_control.utils import rewards
from gymnasium.spaces import Box

class HybridWalk(Walk):
    """ This is a custom class used to experiment with the hybrid walk task for Unitree H1 robot. It works only in this configuration"""
    def __init__(self):
        super().__init__()

    def set_observation_space(self, robot: Robot) -> None:

        self._observation_space = Box(
            low=-np.inf, high=np.inf, shape=(robot.dof * 2 - 1,), dtype=np.float64
        )
        self._observation_space_agent = Box(
            low=-np.inf, high=np.inf, shape=(7 + 6 + robot.lower_body_joints* 2,), dtype=np.float64
        )

        self._action_space_shape_agent = robot.lower_body_joints

    def get_action_space_shape_agent(self):
        return self._action_space_shape_agent

    @property
    def observation_space(self):
        return self._observation_space


    def get_observation_space_agent(self)-> Box:
        return self._observation_space_agent

    def get_obs(self, env: Environment) -> np.array:
        """Override the get_obs method to return only the lower body joints"""
        position = env.data.qpos.flat.copy()[:18] # I removed the upper body joints: 26-8 = 18 
        velocity = env.data.qvel.flat.copy()[:17] # I removed the upper body joints: 25-8 = 17
        state = np.concatenate((position, velocity))
        return state

    def reset_model(self,env: Environment) -> np.array:
        """ Ovveride the reset_model method to reset the reward function with the new get_obs method"""
        self._reward.reset()
        return self.get_obs(env)
