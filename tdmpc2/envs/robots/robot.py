
from abc import ABC, abstractmethod
from ..environment import Environment

class Robot(ABC):

    @abstractmethod
    def update_robot_state(self, env: Environment):
        pass

    