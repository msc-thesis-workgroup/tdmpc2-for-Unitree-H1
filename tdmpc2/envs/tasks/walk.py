
from .task import Task
from ..robots.robot import Robot
from ..environment import Environment
from ..rewards import Reward

import numpy as np
from dm_control.utils import rewards
from gymnasium.spaces import Box

_STAND_HEIGHT = 1.65
_WALK_SPEED = 1

class Walk(Task):
    def __init__(self):
        super().__init__()
        self._observation_space = None
        self.dof = 0 # TODO: Is this necessary?

    def set_observation_space(self, robot: Robot) -> None:
        self._observation_space = Box(
            low=-np.inf, high=np.inf, shape=(robot.dof * 2 - 1,), dtype=np.float64
        )

    @property
    def observation_space(self):
        return self._observation_space

    def get_obs(self, env: Environment) -> np.array:
        position = env.data.qpos.flat.copy()
        velocity = env.data.qvel.flat.copy()
        state = np.concatenate((position, velocity))
        return state

    def unnormalize_action(self, action: np.array) -> np.array:
        # TODO: Check if this is correct
        return (
            2
            * (action - self._env.action_low)
            / (self._env.action_high - self._env.action_low)
            - 1
        )
    
    def normalize_action(self, action: np.array) -> np.array:
        # TODO: Check if this is correct
        return (action + 1) / 2 * (
            self._env.action_high - self._env.action_low
        ) + self._env.action_low
    
    def reset_model(self,env: Environment) -> np.array:
        self._reward.reset()
        return self.get_obs(env)
    
    def get_terminated(self, env: Environment) -> tuple[bool, dict]:
        return env.data.qpos[2] < 0.6, {} # default value was 0.2

    def set_reward(self, reward: Reward):
        self._reward = reward

    def get_reward(self, robot: Robot, action: np.array) -> float:
        return self._reward.get_reward(robot, action)