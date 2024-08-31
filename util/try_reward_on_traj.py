import mujoco
import mujoco_viewer
import numpy as np
import os
from matplotlib import pyplot as plt
import sys
from pyquaternion import Quaternion

# Add the path to the folder containing the module

# Add the path to the top-level directory of the project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tdmpc2')))

from tdmpc2.envs.wrappers.dmc_wrapper import MjDataWrapper, MjModelWrapper
from dm_control.mujoco import index
from dm_control.mujoco.engine import NamedIndexStructs

from tdmpc2.envs.utils import PositionController
from tdmpc2.envs.robots.UnitreeH1 import H1
from tdmpc2.envs.rewards import WalkV4


from util.logger import Logger
#### NOTE:
# To run from terminal: ~/tdmpc2$ python -m util.try_reward_on_traj


def get_joint_torques(ctrl,model,data):

	kp = np.array([200, 200, 200, 300, 40, 200, 200, 200, 300, 40, 300, 100, 100, 100, 100, 100, 100, 100, 100])
	kd = np.array([5, 5, 5, 6, 2, 5, 5, 5, 6, 2, 6, 2, 2, 2, 2, 2, 2, 2, 2])

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

# Configure the simulation
model = mujoco.MjModel.from_xml_path(PATH_TO_MODEL)
data = mujoco.MjData(model)
# Create a viewer
#viewer = mujoco_viewer.MujocoViewer(model, data)

data_wrapper = MjDataWrapper(data)
model_wrapper = MjModelWrapper(model)
axis_indexers = index.make_axis_indexers(model_wrapper)
named = NamedIndexStructs(
	model=index.struct_indexer(model_wrapper, "mjmodel", axis_indexers),
	data=index.struct_indexer(data_wrapper, "mjdata", axis_indexers),
)

####
# Controller 
controller = PositionController()

# Robot and environment
robot = H1()
robot._udate_robot_state_TESTING(model, data)
rewarder = WalkV4(robot=robot)
rewarder.reset()
####
# Logger
header = ["time"] + [f"qpos_{i}" for i in range(model.nq)] + [f"qvel_{i}" for i in range(model.nv)] + [f"qacc_{i}" for i in range(model.nv)] + [f"ctrl_{i}" for i in range(model.na)]
logger = Logger("log.csv", header)

def log_data(model,data,logger):
	message = {
        "time": data.time,
        **{
            f"qpos_{i}": data.qpos[i] for i in range(model.nq)
        },
        **{
            f"qvel_{i}": data.qvel[i] for i in range(model.nv)
        },
        **{
            f"qacc_{i}": data.qacc[i] for i in range(model.nv)
        },
        **{
            f"ctrl_{i}": data.ctrl[i] for i in range(model.na)
        },
    }
	logger.log(message)


# Load the trajectory and configure the simulation
traj = np.load("/home/davide/tdmpc2/util/traj.npy")
horizon = 5
n_step = traj.shape[0]//horizon

for i in range(n_step):
	
	time_traj = traj[i*horizon][0] # time of the trajectory
	target_q = traj[(i+1)*horizon][1:27]
	target_vel = traj[(i+1)*horizon][27:52]
	torque = controller.control_step(model=model, data=data, desired_q_pos=target_q[7:26], desired_q_vel=np.zeros_like(target_q[7:26]))
	
	# data.ctrl[:] = torque
	# #data.qpos[:len(target_q)] = target_q
	# mujoco.mj_step(model, data)

	data.qpos = target_q
	data.qvel = target_vel
	mujoco.mj_forward(model, data)
	#print("reward:", rewarder.get_reward(robot, np.zeros_like(torque)))
	error = data.qpos[7:26] - target_q[7:26]
	log_data(model=model,data=data,logger=logger)

	#print("target_q[0:3]", target_q[0:3])
	#print("dtarget_vel[0:3]", target_vel[0:3])
	# plot the error
	
	
	# plt.stem(error)
	# update_plot(error, data.time)
	# plt.pause(0.0001)
	# input("Press Enter to continue...")
	#viewer.render()
DATA_DIR = "./results/"
logger.plot_columns(DATA_DIR+ f"qpos_iter{i}.png", columns_names=[f"qpos_{i}" for i in range(model.nq)])# references = [target_q[i] for i in range(model.nq)])
        
#print("FINAL qpos", data.qpos)