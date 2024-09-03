
import numpy as np
from .reward import Reward
from dm_control.utils import rewards
from ..robots.robot import Robot
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

import os

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.65 # The ideal height of the head is 1.68 m.

class WalkV5(Reward):

    def __init__(self,robot: Robot):
        super().__init__()
        print("[DEBUG basic_locomotion_tasks]: WalkV5.__init__()")
        self._stand_height = _STAND_HEIGHT
        self._move_speed_lower_bound = 0.9 #0.83 = 3 km/h # 2 km/h = 0.55 
        self._move_speed_upper_bound = 1.25 

        self.ideal_orientation = Quaternion(np.array([1, 0, 0, 0]))

        # load trajectory
        # get path of the file and join it with the file name: mocap_trajectory.npy
        path = os.path.join(os.path.dirname(__file__), "mocap_trajectory.npy")
        self.trajectory = np.load(path, allow_pickle=True)
        print("[DEBUG basic_locomotion_tasks]: self.trajectory.shape:", self.trajectory.shape)
        self.index = 0 

    def set_stand_height(self, stand_height):
        self._stand_height = stand_height
    
    def reset(self) -> None:
        self.index = 0
        
    def get_reward(self, robot: Robot, action: np.ndarray) -> float:

        standing = rewards.tolerance(
            robot.head_height(),
            bounds=(self._stand_height, float("inf")),
            margin=self._stand_height / 2,
        )

        upright = rewards.tolerance(
            robot.torso_upright(),
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        stand_reward = standing * upright

        velocity_x = robot.center_of_mass_velocity()[0] # robot.robot_velocity()[0] # I take only the x component of the velocity.
        #print("[DEBUG basic_locomotion_tasks]: robot.center_of_mass_velocity():", robot.center_of_mass_velocity()[0])
        position = robot.robot_position() # I take only the y component of the position.

        position_y = position[1]

        # smooth start
        if self.index < 15: # 15*0.01 = 0.15 s
            move = 1 - self.index/15 
        else:
            move = rewards.tolerance(
                velocity_x, 
                bounds=(self._move_speed_lower_bound, self._move_speed_upper_bound),
                margin=self._move_speed_lower_bound/2, #
                value_at_margin=0.1,
                sigmoid="gaussian",
            ) 

        centered_reward = rewards.tolerance(
            position_y,
            bounds=(-0.3, 0.3),
            margin=0.3,
            value_at_margin=0.1,
            sigmoid="linear",
        )
        

        orientation_quat = robot.robot_orientation()
        rotation = R.from_quat(orientation_quat.tolist(), scalar_first=True)
        angles = rotation.as_euler('zyx', degrees=True)
        angle_x = angles[0]

        stay_inline_reward = rewards.tolerance(
            angle_x,
            bounds=(-10, 10),
            margin=15,
            value_at_margin=0.1,
            sigmoid="linear",
        )

        move = (move*centered_reward + move*stay_inline_reward)/2

        qpos = robot.get_qpos()[7:26]
        error = qpos - self.trajectory[self.index][7:26]
        error = np.mean(np.abs(error))
        joint_traj_reward = rewards.tolerance(
            error,
            bounds=(0, 0.02),
            margin=0.2,
            value_at_margin=0.1,
            sigmoid="linear",
        )

        self.index += 1
        
        #print("[DEBUG basic_locomotion_tasks]: standing:", standing, "upright:", upright, "stand_reward:", stand_reward, "move:", move, "joint_traj_reward:", joint_traj_reward, "centered_reward:", centered_reward, "stay_inline_reward:", stay_inline_reward)
        reward = stand_reward*(move + joint_traj_reward)/2
        return reward, {
            "stand_reward": stand_reward,
            "small_control": joint_traj_reward,
            "move": move,
            "standing": standing,
            "upright": upright,
        }
