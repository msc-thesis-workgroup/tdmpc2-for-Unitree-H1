
import numpy as np
from crowdsourcing import CostFunction, Sources, TargetBehavior, CrowdSourcing

import os
import sys
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
import warnings
warnings.filterwarnings("ignore")
import hydra
import imageio
import numpy as np
import torch
from termcolor import colored

from tdmpc2.common.parser import parse_cfg
from tdmpc2.common.seed import set_seed
from tdmpc2.envs import make_env
from tdmpc2.tdmpc2 import TDMPC2

from scipy.stats import multivariate_normal


class QuadraticCostFunction(CostFunction):
    def cost_function(x: np.array):
        return np.dot(x, x)

# Create the sources
class Sources(Sources):

    def __init__(self):
        self.sources = []
        self.upper_joint_limits = np.array([0.43, 0.43, 2.53, 2.05, 0.52, 0.43, 0.43, 2.53, 2.05, 0.52, 2.35, 2.87, 3.11, 4.45, 2.61, 2.87, 0.34, 1.3, 2.61])
        self.lower_joint_limits = np.array([-0.43, -0.43, -3.14, -0.26, -0.87, -0.43, -0.43, -3.14, -0.26, -0.87, -2.35, -2.87, -0.34, -1.3,  -1.25, -2.87, -3.11, -4.45, -1.25])
        
    def return_sources(self, state: np.array):
        
        return self.sources

    def get_relative_state_space_dimension(self):
        pass

    @property
    def lower_bound(self):
        lower_bound_limits = np.concatenate((np.zeros(7),self.lower_joint_limits,np.zeros(25)))
        return lower_bound_limits
    
    @property
    def upper_bound(self):
        max_velocity = 5
        upper_bound_limits = np.concatenate((np.zeros(7),self.upper_joint_limits,np.full(25,max_velocity)))
        return upper_bound_limits


    def sample_points(self,xStart,i:int = 0,num_points: int = 1):
        """Samples points from the sources."""
        pass
        
        

    def add_source_from_action(self, action: np.array):
        # I want to create a gaussian source centered in the action and with a certain variance 
        self.sources.append(self.create_multivariate_gaussian_function(action, 0.1))

    def create_multivariate_gaussian_function(self,action:np.array, variance: float): # TODO: Consider moving this to a utility class with many other typical functions
        """
        Create a multivariate Gaussian function centered at the given action with the specified variance.

        Parameters:
        - action: A NumPy array of length 19, serving as the mean of the multivariate Gaussian distribution.
        - variance: The variance for the Gaussian distribution, applied uniformly across all dimensions.

        Returns:
        - A function that calculates the multivariate Gaussian distribution's PDF for a given input x.
        """
        # Create a covariance matrix with the specified variance along the diagonal
        covariance_matrix = np.eye(len(action)) * variance
        
        # Define the multivariate normal distribution
        mv_normal = multivariate_normal(mean=action, cov=covariance_matrix)
        
        # Return the PDF function of the distribution
        return mv_normal.pdf

# Create the target behavior
class TargetBehavior(TargetBehavior):
    def target_behavior():
        return 0

def load_agents(cfg):
    assert os.path.exists(cfg.crowdsource_sources_path), f"Crowdsource Sources Path: {cfg.crowdsource_sources_path} not found! Must be a valid filepath."
    
    agents = []
    torch_files = [f for f in os.listdir(cfg.crowdsource_sources_path) if f.endswith('.pt')]
    
    for file in torch_files:
        try:
            agent = TDMPC2(cfg)
            path = os.path.join(cfg.crowdsource_sources_path, file)
            agent.load(path)  # Load the agent model
            agents.append(agent)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return agents

def action2state(action_list: list, env):
    assert len(action_list) > 0, "Action list must not be empty."
    next_states = []
    for action in action_list:
        next_states.append(env.mock_next_state(action))
    return next_states


@hydra.main(config_name="config", config_path=".")
def run(cfg: dict):
    # Create the cost function

    assert torch.cuda.is_available()
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    action_list = []
    agents = []

    # Make environment
    env = make_env(cfg)
    print("env created. action_space,", env.action_space,"observation_space: ",env.observation_space,"  max_episode_steps: ", env.max_episode_steps)

    # Load agents
    agents = load_agents(cfg)

    print("agents loaded: ", len(agents))

    print(colored(f"Evaluating agent using crowdsourcing algorithm on {cfg.task}:", "yellow", attrs=["bold"]))
    
    if cfg.save_video:
        video_dir = os.path.join(cfg.work_dir, "crowdsourcing_videos")
        os.makedirs(video_dir, exist_ok=True)
    
    task_idx = None
    ep_rewards, ep_successes = [], []
    for i in range(cfg.eval_episodes):
        obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0
        #Adapt to the new observation format
        obs = obs[0] if isinstance(obs, tuple) else obs
        if cfg.save_video:
            frames = [env.render()]
        while not done:

            
            ########            
            action_list.clear()
            for agent in agents:
                action_list.append(agent.act(obs, t0=t == 0, task=task_idx))
            
            next_state_list = action2state(action_list,env)
            # I need to create a function that returns the sources
            sources = Sources()
            sources.add_source_from_action(next_state_list)

            # I need to create a function that returns the target behavior
            target_behavior = TargetBehavior()
            target_behavior.target_behavior()

            cost_function = QuadraticCostFunction()

            # Create the crowd sourcing object
            crowd_sourcing = CrowdSourcing(cost_function, sources, target_behavior)
            x = np.zeros(19)
            best_source_id = crowd_sourcing.execute_greedy(x)
            action = action_list[best_source_id]
            

            ########
            obs, reward, terminated, truncated, info = env.step(action)
            
            #Adapt to the new observation and done format
            obs = obs[0] if isinstance(obs, tuple) else obs
            done = terminated or truncated
            ep_reward += reward
            t += 1
            if cfg.save_video:
                frames.append(env.render())
        ep_rewards.append(ep_reward)
        ep_successes.append(info["success"])
        if cfg.save_video:
            imageio.mimsave(
                os.path.join(video_dir, f"{task}-{i}.mp4"), frames, fps=15
            )
    ep_rewards = np.mean(ep_rewards)
    ep_successes = np.mean(ep_successes)
    



    # Create the crowd sourcing object
    crowd_sourcing = CrowdSourcing(QuadraticCostFunction, Sources, TargetBehavior)
    x = np.zeros(19) 
    crowd_sourcing.execute_greedy(x)


if __name__ == "__main__":
    run()