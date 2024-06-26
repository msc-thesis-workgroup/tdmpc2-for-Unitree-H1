import numpy as np
import math

from abc import ABC, abstractmethod
# Applying the strategy design pattern

class CostFunction(ABC):
    """Abstract class for cost functions."""
    @abstractmethod
    def cost_function(x):
        pass
class Sources(ABC):
    """Abstract class to define the policy sources."""
    @abstractmethod
    def return_sources():
        pass

class TargetBehavior(ABC):
    """Abstract class to define the target behavior."""

    # TODO(all): Refine the goal of this class
    @abstractmethod
    def target_behavior():
        pass        

def return_sources():
    """Returns all the behavior of the sources"""
    pass


class CrowdSourcing:

    def __init__(self, cost_function: CostFunction, sources: Sources, target_behavior: TargetBehavior):
        self.cost_function = cost_function
        self.sources = sources
        self.target_behavior = target_behavior


    def cost_function_time_varying(self,x, k):
        """ Selects the cost function based on the time step k."""
        pass

    
