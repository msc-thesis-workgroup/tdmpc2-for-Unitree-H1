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
from tdmpc2 import TDMPC2

import copy
import pandas as pd
torch.backends.cudnn.benchmark = True
from pyquaternion import Quaternion

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
            if cfg.generalize_movement:
                obs = generalize_walk_direction(obs)
                
            if cfg.save_video:
                frames = [env.render()]
            while not done:
                action = agent.act(obs, t0=t == 0, task=task_idx)
                obs, reward, terminated, truncated, info = env.step(action)
                
                #Adapt to the new observation and done format
                obs = obs[0] if isinstance(obs, tuple) else obs

                if cfg.generalize_movement:
                    obs = generalize_walk_direction(obs)
                    

                done = terminated or truncated
                ep_reward += reward
                t += 1
                if cfg.save_video:
                    frames.append(env.render())
            ep_rewards.append(ep_reward)
            ep_successes.append(info["success"])
            if cfg.save_video:

                df_observations.to_csv( os.path.join(video_dir, f"{task}-{i}.csv"), index=False)

                imageio.mimsave(
                    os.path.join(video_dir, f"{task}-{i}.mp4"), frames, fps=15
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

target_orientation = Quaternion(np.array([1,0,0,0]))
target_position = np.array([0,0,0.98])

home_orientation = Quaternion(np.array([0,0,0,1])) # qpos0 [3:7]
home_position = np.array([0,0,0.98]) # qpos0 [0:3]

# Calculate the transformation matrix from the home orientation to the target orientation
transformation_quat = target_orientation * home_orientation.inverse
transformation_matrix = np.eye(4)
transformation_matrix[0:3,0:3] = transformation_quat.rotation_matrix
transformation_matrix[0:3,3] = np.ones(3)*(target_position - home_position)
transformation_matrix[3,3] = 1
print("transformation_matrix: ", transformation_matrix)
print("transformation_quat: ", transformation_quat.rotation_matrix)
#transformation_matrix[3,0:3] = 0 # It is already it is not necessary to set it to zero 

df_observations = pd.DataFrame(columns=['lin_pos','quat_pos','lin_vel','ang_vel'])

global_frame = Quaternion(np.array([1,0,0,0]))
global_frame = global_frame.rotation_matrix

# def generalize_walk_direction(obs):

#     global df_observations

#     #print("pose_s:", obs[0:7].numpy(),"vel:", obs[26:32].numpy())
    
#     # Adapt to the new observation format
#     current_quat = Quaternion(obs[3:7].numpy())  # Convert tensor slice to numpy array for Quaternion
#     current_position = obs[0:3].numpy()  # Convert tensor slice to numpy array for Quaternion

#     #df_observations.loc[len(df_observations)] = [current_position, current_quat.q, obs[26:29].numpy(), obs[29:32].numpy()]
    
#     # Convert the position vector to a quaternion with zero scalar part
#     # pos_quat = Quaternion(np.array([0] + current_position.tolist()))
    
    
#     # # Apply the rotation
#     # rotated_pos_quat = transformation_quat * pos_quat * transformation_quat.inverse

#     # # Extract the rotated vector from the quaternion
#     # #print("rotated_pos_quat: ", rotated_pos_quat.q)
#     # new_pos = np.array(rotated_pos_quat.q[1:])



#     #new_pos = transformation_quat.rotate(current_position)

#     new_pos = np.dot(transformation_matrix, np.array([current_position.tolist() + [1]]).T)
#     new_pos = new_pos[0:3].T
#     # Calculate the new walk direction
#     new_quat = transformation_quat * current_quat
    



#     local_to_global_quat = Quaternion(np.array([0,0,0,1]))*current_quat.inverse
#     local_to_global_rot = local_to_global_quat.rotation_matrix
#     local_to_global_transformation = np.eye(4)
#     local_to_global_transformation[0:3,0:3] = local_to_global_rot
#     local_to_global_transformation[0:3,3] = current_position
#     local_to_global_transformation[3,3] = 1


#     global_to_new_local_quat = new_quat*Quaternion(np.array([0,0,0,1])).inverse
#     global_to_new_local_rot = global_to_new_local_quat.rotation_matrix
#     global_to_new_local_transformation = np.eye(4)
#     global_to_new_local_transformation[0:3,0:3] = global_to_new_local_rot
#     global_to_new_local_transformation[0:3,3] = -new_pos #????
#     global_to_new_local_transformation[3,3] = 1
    




#     # Convert new_quat.q and new_pos (numpy arrays) to tensors and assign them back to obs
#     obs[0:3] = torch.from_numpy(new_pos).type_as(obs)
#     obs[3:7] = torch.from_numpy(new_quat.q).type_as(obs)

#     lin_vel = obs[26:29].numpy()
#     lin_vel = np.array(transformation_quat.rotate(lin_vel))
#     obs[26:29] = torch.from_numpy(lin_vel).type_as(obs)

#     ang_vel = obs[29:32].numpy()

#     ang_vel_tilde = np.array(ang_vel.tolist() + [1])
#     new_ang_vel_tilde = np.dot(global_to_new_local_transformation, np.dot(local_to_global_transformation, ang_vel_tilde))
    


#     ang_vel = new_ang_vel_tilde[0:3]
#     obs[29:32] = torch.from_numpy(ang_vel).type_as(obs)
#     #print("pose_f:", obs[0:7].numpy(), obs[26:32].numpy())
#     return obs


# compute the inverse of the rotation matrix

def generalize_walk_direction(obs):
    global transformation_quat
    global df_observations

    # Adapt to the new observation format
    current_quat = Quaternion(obs[3:7].numpy())  # Convert tensor slice to numpy array for Quaternion
    current_position = obs[0:3].numpy() # Convert tensor slice to numpy array for Quaternion


    new_quat = transformation_quat * current_quat
    new_pos = transformation_quat.rotate(current_position)

    new_vel = transformation_quat.rotate(obs[26:29].numpy())
    new_ang_vel = transformation_quat.rotate(obs[29:32].numpy())


    df_observations.loc[len(df_observations)] = [new_pos, new_quat.q, new_vel, new_ang_vel]


    obs[0:3] = torch.from_numpy(new_pos).type_as(obs)
    obs[3:7] = torch.from_numpy(new_quat.q).type_as(obs)
    obs[26:29] = torch.from_numpy(new_vel).type_as(obs)
    #obs[29:32] = torch.from_numpy(new_ang_vel).type_as(obs)

    return obs











def generalize_walk_action(action):
    temp = action
    action_copy = action.numpy()
    action = copy.deepcopy(action_copy)

    action[0] = action_copy[5]
    action[1] = action_copy[6]
    action[2] = action_copy[7]
    action[3] = action_copy[8]
    action[4] = action_copy[9]
    action[5] = action_copy[0]
    action[6] = action_copy[1]
    action[7] = action_copy[2]
    action[8] = action_copy[3]
    action[9] = action_copy[4]

    action[11] = action_copy[15]
    action[12] = action_copy[16]
    action[13] = action_copy[17]
    action[14] = action_copy[18]
    action[15] = action_copy[11]
    action[16] = action_copy[12]
    action[17] = action_copy[13]
    action[18] = action_copy[14]

    # convert to tensor
    action = torch.from_numpy(action).type_as(temp)

    return action

 
if __name__ == "__main__":
    evaluate()
