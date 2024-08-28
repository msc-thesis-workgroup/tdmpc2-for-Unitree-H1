# Description: Custom environment for Humanoid tasks, to create custom reward functions. This code is a modified version of the code in the HumanoidBench repository.
import os
import sys

import gymnasium as gym

DEFAULT_EPISODE_STEPS = 1000


from gymnasium.envs import register
from envs.humanoid_robot_env import ROBOTS, TASKS, REWARDS


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

            for reward_name in REWARDS.keys():
                if task in reward_name:
                    version = reward_name.split("-")[1]
                    kwargs["version"] = version
                    env_id = f"{robot}-{task}-{version}"
                    print(f"Registering environment: {env_id} with kwargs: {kwargs}")
                    
                    if "max_episode_steps" in cfg:
                        max_episode_steps = cfg.max_episode_steps
                    else:
                        max_episode_steps = DEFAULT_EPISODE_STEPS
                    register(
                        id=f"{robot}-{task}-{version}",
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

    print("[DEBUG humanoid_locomotion_wrapper.py]: cfg.task:", cfg.task)
    
    strings = cfg.task.split("-")
    assert len(strings) == 3, "Task name must be in the format robot-task-version"
    assert strings[0] in ROBOTS, f"Invalid robot name: {strings[0]}"
    assert strings[1] in TASKS, f"Invalid task name: {strings[1]}"
    versions = [key.split("-")[1] for key in REWARDS.keys()]
    assert strings[2] in versions, f"Invalid reward version: {strings[2]}"
    kwargs = {
        "robot": strings[0],
        "task": strings[1],
        "version": strings[2]
    }
              
    if "frame_skip" in cfg:
        env = gym.make(id=cfg.task, frame_skip=cfg.frame_skip, **kwargs)
    else:
        env = gym.make(id=cfg.task, **kwargs)

    # get the max_episode_steps from the cfg if it is available
    if "max_episode_steps" in cfg:
        print("[DEBUG humanoid_locomotion_wrapper.py]: setting max_episode_steps to", cfg.max_episode_steps)
        env.max_episode_steps = cfg.max_episode_steps
    
    env.max_episode_steps = env.get_wrapper_attr("_max_episode_steps") #TODO: check if this is correct
    print("[DEBUG humanoid_locomotion_wrapper.py]: max_episode_steps:", env.max_episode_steps)
    return env
