import os
import sys

if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"

os.environ["LAZY_LEGACY_OP"] = "0"
import warnings

warnings.filterwarnings("ignore")
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from agents.tdmpc2.tdmpc2 import TDMPC2

from agents.tdmpc2.trainer.offline_trainer import OfflineTrainer
from agents.tdmpc2.trainer.online_trainer import OnlineTrainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True


@hydra.main(config_name="config", config_path=".")
def train(cfg: dict):
    """
    Script for training a single-task H1 robot with TD-MPC2. This is an adaptation of the original training script of TD-MPC2.

    """
    assert torch.cuda.is_available()
    assert cfg.steps > 0, "Must train for at least 1 step."
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.work_dir)
    #print(colored(f"Checkpoint: {cfg.checkpoint}", "blue", attrs=["bold"]))

    # Select trainer
    trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer
    print("Using trainer:", trainer_cls.__name__)
    
    # Create environment
    env=make_env(cfg)

    # Create agent
    agent=TDMPC2(cfg)
    print("Agent created")

    # Create logger with the correct experiment name
    experiment_name = cfg.experiment_name

    logger = Logger(cfg, experiment_name)
    buffer = Buffer(cfg)
    
    if cfg.checkpoint != "None" and os.path.exists(cfg.checkpoint):

        print("Loading checkpoint...")
        agent.load(cfg.checkpoint)
        cfg.from_scratch = False

        # Load the data produced by the previous run
        dir_path = os.path.dirname(cfg.checkpoint)
        #pickle_path = os.path.join(dir_path, "buffer.pkl")
        old_csv = os.path.join(dir_path, "eval.csv")
        step_number = int(cfg.checkpoint.split("-")[-1].split(".")[0]) if cfg.checkpoint != "None" else 0
        
        print("Loaded checkpoint from",cfg.checkpoint)
    else:
        print("No checkpoint provided, training from scratch")
        old_csv = None
        step_number = 0
        cfg.from_scratch = True


    # Create trainer
    trainer = trainer_cls(
        cfg=cfg,
        env=env,
        agent=agent,
        buffer=buffer,
        logger=logger,
    )

    print("Starting training from step", step_number)
    trainer.set_step(step_number,old_csv=old_csv)
 
    if cfg.checkpoint != "None" and os.path.exists(cfg.checkpoint):
        dir_path = os.path.dirname(cfg.checkpoint)
        pickle_path = os.path.join(dir_path, "buffer.pkl")
        if os.path.exists(pickle_path):
            trainer.load_buffer(pickle_path) # TODO(my-rice): what happens if the file does not exist?


        #TODO(my-rice): Note that if you load the buffer from a checkpoint, it will use the cfg file of the previous run. Not the one you are using now.
        # This is not a problem if you don't change the cfg file between runs, but if you do, you will have to find a way to manage the inconsistency. 
        # -> A solution can be check if there are some differences between the two cfg files and if there are, raise an error.        

        else:
            print("WARNING: Buffer not found, it can be a problem for the training. Check the path",pickle_path)

    trainer.train()
    print("\nTraining completed successfully")


if __name__ == "__main__":
    train()
