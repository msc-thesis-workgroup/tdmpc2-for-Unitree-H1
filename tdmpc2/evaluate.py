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

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env

from agents.tdmpc2.tdmpc2 import TDMPC2

import pandas as pd
torch.backends.cudnn.benchmark = True
from pyquaternion import Quaternion

df_info = pd.DataFrame(columns=[
    "stand_reward",
    "small_control",
    "move",
    "standing",
    "upright",
    "angle_x",
    "com_velocity_x",
    "centered_reward",
    "stay_inline_reward",
    "robot_velocity"
])

@hydra.main(config_name="config", config_path=".")
def evaluate(cfg: dict):
    """
    Script for evaluating a single-task / multi-task TD-MPC2 checkpoint.

    Most relevant args:
            `task`: task name (or mt30/mt80 for multi-task evaluation)
            `model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
            `checkpoint`: path to model checkpoint to load
            `eval_episodes`: number of episodes to evaluate on per task (default: 10)
            `save_video`: whether to save a video of the evaluation (default: True)
            `seed`: random seed (default: 1)

    See config.yaml for a full list of args.

    Example usage:
    ````
            $ python evaluate.py task=mt80 model_size=48 checkpoint=/path/to/mt80-48M.pt
            $ python evaluate.py task=mt30 model_size=317 checkpoint=/path/to/mt30-317M.pt
            $ python evaluate.py task=dog-run checkpoint=/path/to/dog-1.pt save_video=true
    ```
    """
    assert torch.cuda.is_available()
    assert cfg.eval_episodes > 0, "Must evaluate at least 1 episode."
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored(f"Task: {cfg.task}", "blue", attrs=["bold"]))
    print(
        colored(
            f'Model size: {cfg.get("model_size", "default")}', "blue", attrs=["bold"]
        )
    )
    print(colored(f"Checkpoint: {cfg.checkpoint}", "blue", attrs=["bold"]))
    if not cfg.multitask and ("mt80" in cfg.checkpoint or "mt30" in cfg.checkpoint):
        print(
            colored(
                "Warning: single-task evaluation of multi-task models is not currently supported.",
                "red",
                attrs=["bold"],
            )
        )
        print(
            colored(
                "To evaluate a multi-task model, use task=mt80 or task=mt30.",
                "red",
                attrs=["bold"],
            )
        )

    # Make environment
    env = make_env(cfg)
    print("env created. action_space,", env.action_space,"observation_space: ",env.observation_space,"  max_episode_steps: ", env.max_episode_steps)

    print("env.model.opt.timestep", env.model.opt.timestep)
    

    # Load agent
    agent = TDMPC2(cfg)
    assert os.path.exists(
        cfg.checkpoint
    ), f"Checkpoint {cfg.checkpoint} not found! Must be a valid filepath."
    agent.load(cfg.checkpoint)


    # Evaluate
    if cfg.multitask:
        print(
            colored(
                f"Evaluating agent on {len(cfg.tasks)} tasks:", "yellow", attrs=["bold"]
            )
        )
    else:
        print(colored(f"Evaluating agent on {cfg.task}:", "yellow", attrs=["bold"]))
    if cfg.save_video:
        video_dir = os.path.join(cfg.work_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
    scores = []
    tasks = cfg.tasks if cfg.multitask else [cfg.task]
    for task_idx, task in enumerate(tasks):
        if not cfg.multitask:
            task_idx = None
        ep_rewards, ep_successes = [], []
        for i in range(cfg.eval_episodes):
            obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0
            #Adapt to the new observation format
            obs = obs[0] if isinstance(obs, tuple) else obs
            # if cfg.generalize_movement:
            #    obs = generalize_walk_direction(obs)
                
            if cfg.save_video:
                frames = [env.render()]
            while not done:
                action = agent.act(obs, t0=t == 0, task=task_idx)

                # record_data(env,action)

                obs, reward, terminated, truncated, info = env.step(action)
                
                #Adapt to the new observation and done format
                obs = obs[0] if isinstance(obs, tuple) else obs

                # if cfg.generalize_movement:
                #    obs = generalize_walk_direction(obs)
                    

                done = terminated or truncated
                ep_reward += reward
                t += 1

                # df_info.loc[len(df_info)] = [
                #     info["stand_reward"],
                #     info["small_control"],
                #     info["move"],
                #     info["standing"],
                #     info["upright"],
                #     info["angle_x"],
                #     info["com_velocity_x"],
                #     info["centered_reward"],
                #     info["stay_inline_reward"],
                #     info["robot_velocity"]
                # ]

                if cfg.save_video:
                    frames.append(env.render())
            ep_rewards.append(ep_reward)
            ep_successes.append(info["success"])
            if cfg.save_video:

                #df_observations.to_csv( os.path.join(video_dir, f"{task}-{i}.csv"), index=False)

                df_info.to_csv( os.path.join(video_dir, f"{task}-{i}.csv"), index=False)

                
                print("t:", t,"Saving video to", video_dir)
                imageio.mimsave(
                    os.path.join(video_dir, f"{task}-{i}.mp4"), frames, fps=1/env.unwrapped.dt
                )
        ep_rewards = np.mean(ep_rewards)
        ep_successes = np.mean(ep_successes)
        if cfg.multitask:
            scores.append(
                ep_successes * 100 if task.startswith("mw-") else ep_rewards / 10
            )
        print(
            colored(
                f"  {task:<22}" f"\tR: {ep_rewards:.01f}  " f"\tS: {ep_successes:.02f}",
                "yellow",
            )
        )
    if cfg.multitask:
        print(
            colored(
                f"Normalized score: {np.mean(scores):.02f}", "yellow", attrs=["bold"]
            )
        )


# TODO(my-rice): this code should be moved to a separate file and organized better.

target_orientation = Quaternion(np.array([1,0,0,0]))
target_position = np.array([0,0,0.98])

home_orientation = Quaternion(np.array([1, 0, 0, 0])) # qpos0 [3:7]
home_position = np.array([0,0,0.98]) # qpos0 [0:3]

# Calculate the transformation matrix from the home orientation to the target orientation
transformation_quat = target_orientation * home_orientation.inverse

df_observations = pd.DataFrame(columns=['lin_pos','quat_pos'])



from torch import Tensor
def generalize_walk_direction(obs):
    global transformation_quat
    global df_observations

    # Adapt to the new observation format
    current_quat = Quaternion(obs[3:7].numpy())  # Convert tensor slice to numpy array for Quaternion
    current_position = obs[0:3].numpy() # Convert tensor slice to numpy array for Quaternion


    new_quat = transformation_quat * current_quat
    new_pos = transformation_quat.rotate(current_position)

    new_vel = transformation_quat.rotate(obs[26-8:29-8].numpy())
    #new_ang_vel = transformation_quat.rotate(obs[29:32].numpy())

    df_observations.loc[len(df_observations)] = [current_position, current_quat]


    obs[0:3] = torch.from_numpy(new_pos).type_as(obs)
    obs[3:7] = torch.from_numpy(new_quat.q).type_as(obs)
    obs[26-8:29-8] = torch.from_numpy(new_vel).type_as(obs)
    #obs[29:32] = torch.from_numpy(new_ang_vel).type_as(obs)

    return obs


if __name__ == "__main__":
    evaluate()
