import mujoco
import mujoco_viewer
import numpy as np
import os
from matplotlib import pyplot as plt
import sys
from pyquaternion import Quaternion
import time
# Add the path to the folder containing the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tdmpc2', 'envs')))


from wrappers.dmc_wrapper import MjDataWrapper, MjModelWrapper
from dm_control.mujoco import index
from dm_control.mujoco.engine import NamedIndexStructs



# from utils import PositionController


def get_joint_torques(ctrl,model,data):

	kp = np.array([200, 200, 200, 300, 40, 200, 200, 200, 300, 40, 300, 100, 100, 100, 100, 100, 100, 100, 100])
	kd = np.array([6, 5, 5, 6, 2, 5, 5, 5, 6, 2, 6, 2, 2, 2, 2, 2, 2, 2, 2])

	actuator_length = data.qpos[7:26] # self.data.actuator_length
	error = ctrl - actuator_length
	m = model
	d = data

	empty_array = np.zeros(m.actuator_dyntype.shape)

	ctrl_dot = np.zeros(m.actuator_dyntype.shape) if np.array_equal(m.actuator_dyntype,empty_array) else d.act_dot[m.actuator_actadr + m.actuator_actnum - 1]

	error_dot = ctrl_dot - data.qvel[6:26] # self.data.actuator_velocity

	joint_torques = kp*error + kd*error_dot

	return joint_torques

PATH_TO_MODEL = "asset/H1/scene.xml"

# Load the model
model = mujoco.MjModel.from_xml_path(PATH_TO_MODEL)
data = mujoco.MjData(model)
# Create a viewer
viewer = mujoco_viewer.MujocoViewer(model, data)
# Set the target qpos
# target_q = np.array([0, 0, 0.98,
# 					1, 0, 0, 0,
# 					0, 0, -0.4, 0.8, -0.4,
# 					0, 0, -0.4, 0.8, -0.4,
# 					0
# 					])

target_q = np.array([0, 0, 0.98,
					1, 0, 0, 0,
					0, 0, -0.4, 0.8, -0.4,
					0, 0, -0.4, 0.8, -0.4,
					0,
					0, 0, 0, 0,
					0, 0, 0, 0])

print("INITIAL qpos", data.qpos)

data_wrapper = MjDataWrapper(data)
model_wrapper = MjModelWrapper(model)
axis_indexers = index.make_axis_indexers(model_wrapper)
named = NamedIndexStructs(
	model=index.struct_indexer(model_wrapper, "mjmodel", axis_indexers),
	data=index.struct_indexer(data_wrapper, "mjdata", axis_indexers),
)

# controller = PositionController(model)
# fig, ax = plt.subplots()
# line, = ax.plot([], [], 'b-')

# def update_plot(error, data_time):
#     line.set_data(np.ones_like(error) * data_time, error)
#     ax.relim()  # Recompute the data limits
#     ax.autoscale_view()  # Rescale the view
#     plt.pause(0.0001)


upper_body_joints = { # The indexes are the indexes of the joints in the qpos array. However, they are hardcoded I found out that in named data structure it should be a dictionary with the name of the joint as key and the index as value. 
    "left_shoulder_roll": 19,
    "left_shoulder_pitch": 18,
    "left_shoulder_yaw": 20,
    "left_elbow": 21,
    "right_shoulder_roll": 23,
    "right_shoulder_pitch": 22,
    "right_shoulder_yaw": 24,
    "right_elbow": 25,
    "torso": 17, 
}

for i in range(1000):

	#torque = controller.control_step(model=model, data=data, desired_q_pos=target_q[7:26], desired_q_vel=np.zeros_like(target_q[7:26]))
	
	# print("data.subtree_linvel", named.data.subtree_linvel)
	# print("data.subtree_linvel left_ankle_link", named.data.subtree_linvel["left_ankle_link"])
	# print("data.subtree_linvel right_ankle_link", named.data.subtree_linvel["right_ankle_link"])
	
	#data.ctrl[:] = torque
	#data.qpos[:len(target_q)] = target_q
	#mujoco.mj_step(model, data)

	# if i % 10 == 0:
	# 	current_quat = Quaternion(data.qpos[3:7])
	# 	q_temp = Quaternion(axis=[0, 0, 1], angle=5*np.pi/180)
	# 	home_orientation = q_temp*current_quat

	# 	data.qpos[3:7] = home_orientation.q 

	data.qpos = target_q
	data.qpos[17] = np.pi/180 * 20 #30 degree = 0.52359877559 #45 degree = 0.78539816339
	#90 degree = 1.57079632679
	#error = data.qpos[7:18] - target_q[7:18]
	mujoco.mj_forward(model, data)
	# plot the erro
	
	#plt.stem(error)
	# update_plot(error, data.time)
	# plt.pause(0.0001)
	#input("Press Enter to continue...")
	viewer.render()

print("FINAL qpos", data.qpos)