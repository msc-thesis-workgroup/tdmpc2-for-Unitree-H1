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

    @abstractmethod
    def get_relative_state_space_dimension():
        pass

    @abstractmethod
    def sample_points(self,xStart,i:int = 0,num_points: int = 1):
        """Samples points from the sources."""
        pass

class TargetBehavior(ABC):
    """Abstract class to define the target behavior."""
    # TODO(all): Refine the goal of this class
    @abstractmethod
    def target_behavior():
        pass        


class CrowdSourcing:

    def __init__(self, cost_function: CostFunction, sources: Sources, target_behavior: TargetBehavior):
        self.cost_function = cost_function
        self.sources = sources
        self.target_behavior = target_behavior


    def cost_function_time_varying(self,x, k: int):
        """ Selects the cost function based on the time step k."""
        
        # for now we are using the same cost function for all time steps
        return self.cost_function.cost_function(x)
    
    # def execute(self, xStart):
    #     """Executes the crowd sourcing algorithm."""
        
    #     #timeHor = list(range(tHor))
    #     #timeHor.reverse()
    #     S = self.sources.return_sources() # As the notation in the paper

    #     xDim = self.sources.get_relative_state_space_dimension() # TODO(): xdim dovrebbe essere l'observation space
    #     # Initialize the relative state space

    #     rHat = [0]*xDim #Reward modifier, same dimension as the relative state space
    #     weights = np.zeros((xDim, len(S))) # := the vector 'a' in the paper. In reality it is a matrix with xDim rows and S columns

    #     #if tHor == 0: # If the time horizon is 0 timeHor would be an empty list.
    #     #    timeHor = [0]
        
    #     #for k in timeHor:
            
    #         # If I am correct, now we have to loop over all the sources and 
    #         # sample 20 points for each source Si and compute the next state (x_k) applying the monte carlo method
            

    #         # Quesito CRUCIALE: l'algoritmo prevede di partire da N (timeHor-1) e poi scendere fino a 1 (0), ma non è chiaro come si possa fare
    #         # dato che ho solo lo stato attuale(x_k-1). In altre parole come posso ottenere lo stato x_{timeHor -1} se non so come evolve allo stato successivo?
    #         # Devo prendere ogni possibile stato? Ma lo stato è continuo.
    #         # Posso partire da x_k-1 e fare un passo avanti prendendo x_k dato x_k-1 e poi posso continuare fino a x_{timeHor-1}?
    #         # Ho appena controllato EMILAND GARRABE fa così. Anche secondo me è l'unica soluzione.


    #     for i in range(len(S)):
                
                
                
                
    #         x_k = xStart

    #         r_hat[x_k] = -min(weights[x_k])



    #         #act_state = actualState(xStart,x)
    #         rHat[x] = -min(weights[x])
    #         act_state = actualState(xStart,x)
    #         r = np.add(return_reward(act_state), rHat)
    #         sources = return_sources(act_state)
    #         target = return_target(act_state)
    #         #fare un vettore che va da 0 a S e metterci il vettore a e poi estrarre l'indice per cui è minimo
    #         for i in range(S):
    #             weights[x,i] = DKL(sources[i], target) +- np.dot(sources[i], r)
    #     if k==0:
    #         j = np.argmin(weights[0])
    #         return(return_sources(xStart)[j])
    
    # def receding_horizon_DM(self, state, tHor, stateSpace, r):
    #     xInd = np.where(stateSpace == state)[0][0] #Index of the state in the reduced state space

    #     num_sources = self.sources.shape[0]
    #     full_state_space_dim = self.sources.shape[1] 
        
    #     dim = len(stateSpace)
    #     rHat = np.array([0]*full_state_space_dim) #rHat as in the algorithm 
    #     weights = np.zeros((num_sources, dim)) #Decision-making weights

    #     sources = np.zeros((num_sources, dim, full_state_space_dim)) #The sources and target we use correspond to the reduced state space
    #     target = np.zeros((dim,full_state_space_dim))
        
    #     self.PMF_target = behaviors[self.nTarget]
    #     for i in range(dim):
    #         target[i] = self.PMF_target[stateSpace[i]]


    #     for i in range(num_sources):
    #         for j in range(dim):
    #             sources[i,j] = behaviors[i,stateSpace[j]]
    #     if tHor == 0: #Safety as python doesn't like range(0)
    #         tHor = 1
    
    #     timeHor = list(range(tHor))
    #     timeHor.reverse()
    
    #     for t in timeHor:
    #         rBar = r + rHat #Adapt reward
    #         for i in range(num_sources):
    #             for j in range(dim):
    #                 weights[i,j] = self.__DKL(sources[i,j], target[j]) - np.dot(sources[i,j], rBar) #calculate weights
    #         for i in range(dim):
    #             rHat[stateSpace[i]] = -min(weights[:,i]) #Calculate rHat
    #     indMin = np.argmin(weights[:,xInd])
    #     pf = sources[indMin, xInd] #Pick pf
    #     return(np.random.choice(range(full_state_space_dim), p = pf)) #Sample pf and return resulting state

    def execute_greedy(self, x):
        """Executes the greedy algorithm."""
        sources = self.sources.return_sources()
        num_sources = len(sources)
        obs_space = len(x)
        weights = np.zeros((num_sources, obs_space))
        target = self.target_behavior.target_behavior()

        # check if the target is compatible with the observation space

        for i in range(num_sources):
            weights[i] = self.__DKL(sources[i], target) # Is this the correct way to calculate the DKL between two pfs?
            # calculate the expected cost given the state x and the source i

            # To calculate the expected reward we need to calculate the expected cost through the monte carlo method

            n_samples = 20
            samples = self.sources.sample_points(x, i, n_samples)
            # calculate the expected cost
            expected_cost = 0
            for s in samples:
                expected_cost += self.cost_function.cost_function(s)
            expected_cost /= n_samples

            weights[i] -= expected_cost

        # select the source with the minimum weight
        min_index = np.argmin(weights)
        return sources[min_index]
            

        


if __name__ == "__main__":
    # Create the cost function
    class QuadraticCostFunction(CostFunction):
        def cost_function(x: np.array):
            return np.dot(x, x)

    # Create the sources
    class Sources(Sources):
        def return_sources():
            return [1, 2, 3, 4, 5, 6]
        
        def get_relative_state_space_dimension():
            return 5

    # Create the target behavior
    class TargetBehavior(TargetBehavior):
        def target_behavior():
            return 0

    # Create the crowd sourcing object
    crowd_sourcing = CrowdSourcing(QuadraticCostFunction, Sources, TargetBehavior)
    x = np.zeros(19) 
    crowd_sourcing.execute_greedy(x)
    
