
from .task import Task
from ..robots.robot import Robot
from ..environment import Environment
from ..rewards import Reward

import numpy as np
from dm_control.utils import rewards
from gymnasium.spaces import Box

class Walk(Task):
    def __init__(self):
        super().__init__()
        self._observation_space = None
        self.dof = 0 # TODO: Is this necessary?

    def set_observation_space(self, robot: Robot) -> None:
        self._observation_space = Box(
            low=-np.inf, high=np.inf, shape=(robot.dof * 2 - 1,), dtype=np.float64
        )
        self._observation_space_agent = self._observation_space
        self._action_space_shape_agent = robot.joints

    def get_action_space_shape_agent(self):
        return self._action_space_shape_agent
    def get_observation_space_agent(self)-> Box:
        return self._observation_space_agent

    @property
    def observation_space(self):
        return self._observation_space

    def get_obs(self, env: Environment) -> np.ndarray:
        position = env.data.qpos.flat.copy()
        velocity = env.data.qvel.flat.copy()
        state = np.concatenate((position, velocity))
        return state

    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        # TODO: Check if this is correctnd
        return (
            2
            * (action - self._env.action_low)
            / (self._env.action_high - self._env.action_low)
            - 1
        )
    
    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        # TODO: Check if this is correct
        return (action + 1) / 2 * (
            self._env.action_high - self._env.action_low
        ) + self._env.action_low
    
    def reset_model(self,env: Environment) -> np.ndarray:
        self._reward.reset()
        return self.get_obs(env)
    
    def get_terminated(self, env: Environment) -> tuple[bool, dict]:
        return env.data.qpos[2] < 0.7, {} # default value was 0.2

    def set_reward(self, reward: Reward):
        self._reward = reward

    def get_reward(self, robot: Robot, action: np.ndarray) -> float:
        return self._reward.get_reward(robot, action)

def check_collision(env: Environment) -> bool:
    # check collision with the robot

    contacts = [env.data.contact[i] for i in range(env.data.ncon)]

    feet = ["left_ankle_link", "right_ankle_link"]
    feet_ids = [env.model.body(bn).id for bn in feet]
    for i,c in enumerate(contacts):
        geom1_body = env.model.body(env.model.geom_bodyid[c.geom1])
        geom2_body = env.model.body(env.model.geom_bodyid[c.geom2])
        # print("geom1_body",geom1_body)
        # print("geom2_body",geom2_body)
        geom1_is_floor = (env.model.body(geom1_body.rootid).name!="pelvis")
        geom2_is_foot = (env.model.geom_bodyid[c.geom2] in feet_ids)
        if not(geom1_is_floor and geom2_is_foot):
            return True
    return False