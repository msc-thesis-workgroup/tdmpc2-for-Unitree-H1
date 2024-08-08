
from abc import ABC, abstractmethod
import numpy as np
from ..robots.robot import Robot

class Reward(ABC):

    @abstractmethod
    def get_reward(self, robot: Robot, action: np.array) -> float:
        pass