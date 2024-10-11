
import numpy as np
from .reward import Reward
from dm_control.utils import rewards
from ..robots.robot import Robot
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.65 # The ideal height of the head is 1.68 m.

class HybridWalkV0(Reward):

    def __init__(self,robot: Robot):
        super().__init__()
        print("[DEBUG basic_locomotion_tasks]: WalkV0Easy.__init__()")
        self._stand_height = _STAND_HEIGHT
        self._move_speed_lower_bound = 0.83 #0.83 # 3 km/h # 2 km/h = 0.55 
        self._move_speed_upper_bound = 1.5 # 6.4 km/h #4 km/h = 1.11 m/s

        self.reset()
        self.robot = robot

        # self.ideal_quat = Quaternion(1, 0, 0, 0)
        

    def reset(self) -> None:
        pass

    def get_reward(self, robot: Robot, action: np.ndarray) -> float:
        self.robot = robot
        standing = rewards.tolerance(
            robot.head_height(),
            bounds=(self._stand_height, float("inf")),
            margin=self._stand_height/6,
            value_at_margin=0,
            sigmoid="linear",
        )
        upright = rewards.tolerance(
            robot.torso_upright(),
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9, # it doesn't make sense
            value_at_margin=0,
        )
        stand_reward = standing * upright

        ctrl_ranges = robot.get_ctrl_ranges()[0:11]
        actuator_forces = np.abs(robot.actuator_forces()[0:11]) # The ctrl range is symmetric, so I can take the absolute value.
        # Get actuator forces only for the lower body joints
        actuator_forces = actuator_forces/ctrl_ranges[:, 1] # I divide by the maximum value of the control range.
        actuator_forces = np.mean(actuator_forces)
        control_reward = 1 - actuator_forces

        velocity_x = robot.center_of_mass_velocity()[0] # robot.robot_velocity()[0] # I take only the x component of the velocity.
        #print("velocity_x",velocity_x)
        position = robot.robot_position() # I take only the y component of the position.
        position_y = position[1]

        move = rewards.tolerance(
            velocity_x, 
            bounds=(self._move_speed_lower_bound, self._move_speed_upper_bound),
            margin=self._move_speed_lower_bound,
            value_at_margin=0.1,
            sigmoid="gaussian",
        ) 

        centered_reward = rewards.tolerance(
            position_y,
            bounds=(-0.6, 0.6),
            margin=0.4,
            value_at_margin=0.1,
            sigmoid="linear",
        )
        
        # fix torso orientation
        torso_jnt = robot.get_qpos()[17]
        
        stay_inline_reward = rewards.tolerance(
            torso_jnt,
            bounds=(-0.35, 0.35), # -20 degrees, 20 degrees
            margin=0.45, # 25 degrees # (-
            value_at_margin=0.7,
            sigmoid="linear",
        )

        move = (move*centered_reward + move*stay_inline_reward)/2

        move = (5 * move + 1) / 6
        control_reward = (control_reward + 2) / 3
        reward = stand_reward*move*control_reward
        
        #print("move",move,"control_reward",control_reward,"stand_reward",stand_reward)

        return reward, {
            "stand_reward": stand_reward,
            "small_control": control_reward,
            "move": move,
            "standing": standing,
            "upright": upright,
        }
    
