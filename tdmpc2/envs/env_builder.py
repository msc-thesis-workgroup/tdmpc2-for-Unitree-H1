# Description: Custom environment for Humanoid tasks, to create custom reward functions. This code is a modified version of the code in the HumanoidBench repository.
import os
import sys

import numpy as np
import gymnasium as gym

DEFAULT_EPISODE_STEPS = 1000


from gymnasium.envs import register
from envs.humanoid_robot_env import ROBOTS, TASKS


def _test():
    print("Testing humanoid_locomotion_wrapper.py")

def register_envs(cfg):
    """
    This function registers all the environments made available by this repository.
    """

    for robot in ROBOTS:
        for task, task_obj in TASKS.items():
            task_obj = task_obj()
            kwargs = task_obj.kwargs.copy()
            kwargs["robot"] = robot
            kwargs["task"] = task
            if "max_episode_steps" in cfg:
                max_episode_steps = cfg.max_episode_steps
            else:
                max_episode_steps = DEFAULT_EPISODE_STEPS
            register(
                id=f"{robot}-{task}-v0",
                entry_point="envs.humanoid_robot_env:HumanoidRobotEnv",
                max_episode_steps= max_episode_steps,
                kwargs=kwargs,
            )


def make_env(cfg):
    """
    Make Humanoid environment for locomotion task.
    """

    if sys.platform != "darwin" and "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"
    if "SLURM_STEP_GPUS" in os.environ:
        os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_STEP_GPUS"]
        print(f"EGL_DEVICE_ID set to {os.environ['SLURM_STEP_GPUS']}")
    if "SLURM_JOB_GPUS" in os.environ:
        os.environ["EGL_DEVICE_ID"] = os.environ["SLURM_JOB_GPUS"]
        print(f"EGL_DEVICE_ID set to {os.environ['SLURM_JOB_GPUS']}")

    register_envs(cfg)

    # Create the custom environment with the characteristics specified in the register function.
    env = gym.make(cfg.task,)
    #env = HumanoidLocomotionWrapper(env, cfg)

    # get the max_episode_steps from the cfg if it is available
    if "max_episode_steps" in cfg:
        print("[DEBUG humanoid_locomotion_wrapper.py]: setting max_episode_steps to", cfg.max_episode_steps)
        #env.env.max_episode_steps = cfg.max_episode_steps
        env.max_episode_steps = cfg.max_episode_steps
    
    if "frame_skip" in cfg:
        print("[DEBUG humanoid_locomotion_wrapper.py]: setting frame_skip to", cfg.frame_skip)
        #env.env.frame_skip = cfg.frame_skip
        env.frame_skip = cfg.frame_skip

    env.max_episode_steps = env.get_wrapper_attr("_max_episode_steps") #TODO(my-rice): I want to try to use way less episodes. I want to focus on the potential of the reward function. A good reward function should be able to solve the task in a few episodes.
    
    return env
