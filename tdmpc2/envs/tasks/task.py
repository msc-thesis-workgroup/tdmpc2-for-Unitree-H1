from abc import ABC, abstractmethod
import numpy as np

from ..environment import Environment

class Task(ABC):
    kwargs = {}
    @abstractmethod
    def get_reward(self, state: np.array, action: np.array) -> float:
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
    def get_terminated(self, state: np.array) -> bool:
        pass