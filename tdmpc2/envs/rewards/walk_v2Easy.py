
import numpy as np
from .reward import Reward
from dm_control.utils import rewards
from ..robots.robot import Robot
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.65

class WalkV2Easy(Reward):

    def __init__(self,robot: Robot):
        super().__init__()
        self._stand_height = _STAND_HEIGHT
        self._move_speed = 1 # 3.6 km/h
        self._move_speed_lower_bound = 0.55 # 2 km/h #0.83 # 3 km/h
        self._move_speed_upper_bound = 1.11 # 5 km/h #4 km/h = 1.11 m/s

        self.ideal_orientation = Quaternion(np.array([1, 0, 0, 0]))

    def set_stand_height(self, stand_height):
        self._stand_height = stand_height
    
    def reset(self) -> None:
        pass

    def get_reward(self, robot: Robot, action: np.ndarray) -> float:

        standing = rewards.tolerance(
            robot.head_height(),
            bounds=(self._stand_height, float("inf")),
            margin=self._stand_height / 2,
        )
        upright = rewards.tolerance(
            robot.torso_upright(),
            bounds=(0.75, float("inf")), # Lowered the lower bound from 0.9 to 0.75
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
        #small_control = (3 + control_reward) / 4 # I want to give more importance to the other rewards than to the control_reward. It is obvious that the control signal cannot be 0.

        velocity_x = robot.center_of_mass_velocity()[0] # robot.robot_velocity()[0] # I take only the x component of the velocity.
        #print("[DEBUG basic_locomotion_tasks]: robot.center_of_mass_velocity():", robot.center_of_mass_velocity()[0])
        position_y = robot.robot_position()[1] # I take only the y component of the position.

        move = rewards.tolerance(
            velocity_x, 
            bounds=(self._move_speed_lower_bound, self._move_speed_upper_bound),
            margin=self._move_speed/2, #
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

        reward = stand_reward*(3*move + 3*control_reward)/5
        return reward, {
            "stand_reward": stand_reward,
            "small_control": control_reward,
            "move": move,
            "standing": standing,
            "upright": upright,
        }
