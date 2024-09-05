
import numpy as np
from .reward import Reward
from dm_control.utils import rewards
from ..robots.robot import Robot
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.65 # The ideal height of the head is 1.68 m.




class WalkV6(Reward):

    def __init__(self,robot: Robot):
        super().__init__()
        print("[DEBUG basic_locomotion_tasks]: WalkV0Easy.__init__()")
        self._stand_height = _STAND_HEIGHT
        self._move_speed_lower_bound = 0.83 #0.83 # 3 km/h # 2 km/h = 0.55 
        self._move_speed_upper_bound = 1.6 # 5.7 km/h #4 km/h = 1.11 m/s

        self.reset()
        self.robot = robot
        

    def reset(self) -> None:
        pass

    def get_reward(self, robot: Robot, action: np.ndarray) -> float:
        self.robot = robot
        standing = rewards.tolerance(
            robot.head_height(),
            bounds=(self._stand_height, float("inf")),
            margin=self._stand_height/8,
            value_at_margin=0,
            sigmoid="linear",
        )
        upright = rewards.tolerance(
            robot.torso_upright(),
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        stand_reward = standing * upright

        ctrl_ranges = robot.get_ctrl_ranges()
        actuator_forces = np.abs(robot.actuator_forces()) # The ctrl range is symmetric, so I can take the absolute value.
        actuator_forces = actuator_forces/ctrl_ranges[:, 1] # I divide by the maximum value of the control range.
        actuator_forces = np.mean(actuator_forces)
        control_reward = 1 - actuator_forces # I want to penalize the control signal. The reward is 1 minus the mean of the normalized control signal.
        
        velocity_x = robot.center_of_mass_velocity()[0] # robot.robot_velocity()[0] # I take only the x component of the velocity.
        #print("velocity_x",velocity_x)
        position = robot.robot_position() # I take only the y component of the position.

        position_y = position[1]

        move = rewards.tolerance(
            velocity_x, 
            bounds=(self._move_speed_lower_bound, self._move_speed_upper_bound),
            margin=self._move_speed_lower_bound/2, # 0.83
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
            bounds=(-8, 8),
            margin=32,
            value_at_margin=0.1,
            sigmoid="linear",
        )

        q_left_roll = self.robot.get_qpos()[19] # 19 is the index of the left roll joint
        q_right_roll = self.robot.get_qpos()[23] # 23 is the index of the right roll joint
        left_roll_reward = rewards.tolerance(
            q_left_roll,
            bounds=(-0.1,0.1),
            margin=0.5,
            value_at_margin=0.1,
            sigmoid="linear",
        )

        right_roll_reward = rewards.tolerance(
            q_right_roll,
            bounds=(-0.1,0.1),
            margin=0.5,
            value_at_margin=0.1,
            sigmoid="linear",
        )

        arms_reward = (left_roll_reward + right_roll_reward) / 2
        arms_reward = (1 + arms_reward) / 2

        move = (move*centered_reward + move*stay_inline_reward)/2
        move = (5 * move + 1) / 6
        
        control_reward = (control_reward + 2) / 3
        reward = stand_reward*move*control_reward*arms_reward
        
        return reward, {
            "stand_reward": stand_reward,
            "small_control": control_reward,
            "move": move,
            "standing": standing,
            "upright": upright,
        }
