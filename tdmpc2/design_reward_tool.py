from envs import make_env
import hydra

from common.parser import parse_cfg
from common.seed import set_seed
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer

import os
from tdmpc2 import TDMPC2


@hydra.main(config_name="config", config_path=".")
def evaluate(cfg: dict):

    assert cfg.steps > 0, "Must train for at least 1 step."
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    
    # Select trainer
    trainer_cls = OfflineTrainer

    env=make_env(cfg)
    #reward = env.reward_range
    #print(reward)

    task_idx = None

    # Load agent
    agent = TDMPC2(cfg)
    assert os.path.exists(
        cfg.checkpoint
    ), f"Checkpoint {cfg.checkpoint} not found! Must be a valid filepath."
    agent.load(cfg.checkpoint)
    done = False
    obs = env.reset()
    obs = obs[0] if isinstance(obs, tuple) else obs
    t = 0
    ep_reward = 0

    while not done:
        action = agent.act(obs, t0=t == 0, task=task_idx)
        obs, reward, terminated, truncated, info = env.step(action)
        
        #Adapt to the new observation and done format
        obs = obs[0] if isinstance(obs, tuple) else obs
        done = terminated or truncated
        #print("Reward(",t,"): ", reward)
        ep_reward += reward
        t += 1

if __name__ == "__main__":
    # Create environment
    evaluate()    
    
