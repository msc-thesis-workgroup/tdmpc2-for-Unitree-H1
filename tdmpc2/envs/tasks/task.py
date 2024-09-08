from abc import ABC, abstractmethod
import numpy as np
from ..rewards import Reward
from ..robots import Robot
from ..environment import Environment

class Task(ABC):
    kwargs = {}
    @abstractmethod
    def get_reward(self, robot: Robot, action: np.array) -> float:
        pass
    
    @abstractmethod
    def get_obs(self, env: Environment) -> np.array:
        pass

    @abstractmethod
    def unnormalize_action(self, action: np.array) -> np.array:
        pass

    @abstractmethod
    def normalize_action(self, action: np.array) -> np.array:
        pass

    @abstractmethod
    def reset_model(self) -> np.array:
        pass

    @abstractmethod
    def get_terminated(self, env: Environment) -> tuple[bool, dict]:
        pass

    @abstractmethod
    def set_reward(self, reward: Reward):
        pass
        
    @abstractmethod
    def get_action_space_shape_agent(self):
        pass

    @abstractmethod
    def get_observation_space_agent(self):
        pass
