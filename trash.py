from pyquaternion import Quaternion

import numpy as np


obs = np.ones(51)*2
obs[3:7] = np.array([0,0,0,1])
obs[0:3] = np.array([-1,-1,0.98])

obs[26:29] = np.array([-1,-1,0])
obs[29:32] = np.array([0.01,0.02,0.03])

target_orientation = Quaternion(np.array([1,0,0,0]))
target_position = np.array([0,0,0.98])

home_orientation = Quaternion(np.array([0,0,0,1])) # qpos0 [3:7]
#home_orientation = Quaternion(axis=[0, 0, 1], angle=3.14159265 / 2)
home_position = np.array([0,0,0.98]) # qpos0 [0:3]

# Calculate the transformation matrix from the home orientation to the target orientation
transformation_quat = target_orientation * home_orientation.inverse
print("transformation_quat_rotation_matrix: ", transformation_quat.rotation_matrix)


def generalize_walk_direction(obs):
    global transformation_quat

    # Adapt to the new observation format
    current_quat = Quaternion(obs[3:7])  # Convert tensor slice to numpy array for Quaternion
    current_position = obs[0:3] # Convert tensor slice to numpy array for Quaternion


    new_quat = transformation_quat * current_quat
    new_pos = transformation_quat.rotate(current_position)
    #df_observations.loc[len(df_observations)] = [current_position, current_quat.q, obs[26:29].numpy(), obs[29:32].numpy()]

    new_vel = transformation_quat.rotate(obs[26:29])
    new_ang_vel = transformation_quat.rotate(obs[29:32])

    obs[0:3] = new_pos
    obs[3:7] = new_quat.q
    obs[26:29] = new_vel
    obs[29:32] = new_ang_vel

    return obs


print("obs:\n", obs)
new_obs = generalize_walk_direction(obs)
print("new_obs:\n", new_obs)