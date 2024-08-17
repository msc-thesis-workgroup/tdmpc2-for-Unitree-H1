
import numpy as np
from .reward import Reward
from dm_control.utils import rewards
from ..robots.robot import Robot
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.65


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

class WalkV2(Reward):

    def __init__(self,robot: Robot):
        super().__init__()
        self._stand_height = _STAND_HEIGHT
        self._move_speed = 1 # 3.06 km/h
        self._move_speed_lower_bound = 0.83 # 3 km/h
        self._move_speed_upper_bound = 1.11 # 4 km/h 

        upper_limits = robot.get_upper_limits()
        lower_limits = robot.get_lower_limits()

        offset = 7 # the first 7 joints are the root joints

        # Take the indexes of the upper body joints and subtract the offset
        upper_body_joints_idx = np.array(list(upper_body_joints.values())) - offset

        # TODO: Test da fare. Modificare in scene.xml e vedere se cambia qualcosa.
        #     robot._env.model.key_qpos: [[ 0.    0.    0.98  1.    0.    0.    0.    0.    0.   -0.4   0.8  -0.4
        # 0.    0.   -0.4   0.8  -0.4   0.    0.    0.    0.    0.    0.    0.
        # 0.    0.  ]]

        qpos0 = robot._env.model.key_qpos[0]

        self.upper_body_joints_bounds = {}
        for joint_index in upper_body_joints_idx:
            distance = (upper_limits[joint_index] - lower_limits[joint_index])*0.02
            self.upper_body_joints_bounds.update({joint_index+offset : (qpos0[joint_index+offset] - distance, qpos0[joint_index+offset] + distance)})
            #print("index:", joint_index,"ind+off:",joint_index+offset, "upper_limits[index]:", upper_limits[joint_index], "lower_limits[index]:", lower_limits[joint_index], "distance", distance, "qpos0[joint_index+offset]:", qpos0[joint_index+offset])

        # for key, value in self.upper_body_joints_bounds.items():
        #     print("[DEBUG basic_locomotion_tasks]: key:", key, "value:", value)

        
        self.ideal_orientation = Quaternion(np.array([1, 0, 0, 0]))


    def set_stand_height(self, stand_height):
        self._stand_height = stand_height
        
    def get_reward(self, robot: Robot, action: np.array) -> float:

        standing = rewards.tolerance(
            robot.head_height(),
            bounds=(self._stand_height, float("inf")),
            margin=self._stand_height / 4,
        )
        upright = rewards.tolerance(
            robot.torso_upright(),
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        stand_reward = standing * upright

        # I want to compute the reward for small control in the following way:
        ctrl_ranges = robot.get_ctrl_ranges()
        actuator_forces = np.abs(robot.actuator_forces()) # The ctrl range is symmetric, so I can take the absolute value.
        actuator_forces = actuator_forces/ctrl_ranges[:, 1] # I divide by the maximum value of the control range.


        control_reward = 1 - np.mean(actuator_forces) # I want to penalize the control signal. The reward is 1 minus the mean of the normalized control signal.
        small_control = control_reward
        #small_control = (3 + control_reward) / 4 # I want to give more importance to the other rewards than to the control_reward. It is obvious that the control signal cannot be 0.

        reward_upper_body = 0
        joint_position = robot.get_qpos()
        for key, (low,high) in self.upper_body_joints_bounds.items():
            #print("[DEBUG basic_locomotion_tasks]: joint_position[i]:", joint_position[key], "low:", low, "high:", high)
            reward_upper_body += rewards.tolerance(
                joint_position[key],
                bounds=(low, high),
                margin=(high - low) / 3,
                sigmoid="gaussian",
            )
        
        reward_upper_body = reward_upper_body / len(self.upper_body_joints_bounds)

        #reward_upper_body = (1 + 2*reward_upper_body) / 3

        velocity_x = robot.robot_velocity()[0] # I take only the x component of the velocity.
        position_y = robot.robot_position()[1] # I take only the y component of the position.
        #print("[DEBUG basic_locomotion_tasks]: velocity_x:", velocity_x )#, "position_y:", position_y)
        #print("[DEBUG basic_locomotion_tasks]: com_velocity_x:", com_velocity_x)
        move = rewards.tolerance(
            velocity_x, 
            bounds=(self._move_speed_lower_bound, self._move_speed_upper_bound),
            margin=self._move_speed/3,
            value_at_margin=0.1,
            sigmoid="gaussian",
        ) 
        #print("[DEBUG basic_locomotion_tasks]: robot.center_of_mass_velocity():", com_position_y)
        
        centered_reward = rewards.tolerance(
            position_y,
            bounds=(-0.3, 0.3),
            margin=0.2,
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
            margin=10,
            value_at_margin=0.1,
            sigmoid="linear",
        )

        reward = stand_reward*(2*small_control + 5*move + reward_upper_body + centered_reward + stay_inline_reward)/10

        return reward, {
            "stand_reward": stand_reward,
            "small_control": small_control,
            "move": move,
            "standing": standing,
            "upright": upright,
        }
