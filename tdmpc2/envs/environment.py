
from abc import ABC, abstractmethod
import numpy as np 

class Environment(ABC):
    @abstractmethod
    def step(self, action: np.ndarray): # -> tuple[np.array, float, bool, dict]|np.array: # TODO check the output of this method. obs, reward, terminated, info
        pass

    @abstractmethod
    def reset_model(self): # -> np.array:
        pass
    
    @abstractmethod
    def get_obs(self) -> np.ndarray:
        pass

    @abstractmethod
    def seed(self, seed=None) -> None:
        pass

    @abstractmethod
    def render(self) -> None:
        pass