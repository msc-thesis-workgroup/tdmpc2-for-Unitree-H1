
from abc import ABC, abstractmethod
from ..environment import Environment

class Robot(ABC):
    def __init__(self): 
        self.dof = 0
        self.joints = 0
    
    @abstractmethod
    def update_robot_state(self, env: Environment) -> None:
        """Updates the robot state with the environment data."""
        pass

    