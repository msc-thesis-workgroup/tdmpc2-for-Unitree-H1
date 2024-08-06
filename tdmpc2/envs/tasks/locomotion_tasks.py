import os

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.spaces import Box
from dm_control.utils import rewards

# Local import
from .task import Task

from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

import copy
# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.65
_CRAWL_HEIGHT = 0.8

# Horizontal speeds above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 5

upper_body_joints = {
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


class Walk(Task):
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0",
        "h1hand": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "h1touch": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
    }

    # TODO: Ho scoperto che key_qpos è esattamente il qpos0 del modello. TODO: Test da fare. Modificare in scene.xml e vedere se cambia qualcosa.
        #     self.robot._env.model.key_qpos: [[ 0.    0.    0.98  1.    0.    0.    0.    0.    0.   -0.4   0.8  -0.4
        # 0.    0.   -0.4   0.8  -0.4   0.    0.    0.    0.    0.    0.    0.
        # 0.    0.  ]]


    _move_speed = 1.44 # 5.2 km/h
    _move_speed_lower_bound = 1.11 # 4 km/h 
    _move_speed_upper_bound = 1.78 # 6.4 km/h
    #htarget_low = np.array([-10.0, -2.0, 0.8])
    #htarget_high = np.array([1000.0, 2.0, 2.0])
    success_bar = 700

    @property
    def observation_space(self):
        return Box(
            low=-np.inf, high=np.inf, shape=(self.robot.dof * 2 - 1,), dtype=np.float64
        )

    def __init__(self, robot=None, env=None, **kwargs):
        #instanciate the father class
        super().__init__(robot=robot, env=env, **kwargs)
        if robot is None or env is None:
            return
        
        # NOTE(my-rice): WALK viene instanziata n volte. La prima volta senza passare robot. La seconda volta con robot. 
        upper_limits = self.robot.get_upper_limits()
        lower_limits = self.robot.get_lower_limits()

        offset = 7 # the first 7 joints are the root joints

        # Take the indexes of the upper body joints and subtract the offset
        upper_body_joints_idx = np.array(list(upper_body_joints.values())) - offset
        qpos0 = self.robot._env.model.key_qpos[0]
        #print("[DEBUG basic_locomotion_tasks]: qpos0:", qpos0)

        #self.ideal_qpos = np.array([qpos0[idx] for idx in upper_body_joints_idx])
        #print("[DEBUG basic_locomotion_tasks]: ideal_qpos:", self.ideal_qpos)
        self.upper_body_joints_bounds = {}
        for joint_index in upper_body_joints_idx:
            distance = (upper_limits[joint_index] - lower_limits[joint_index])*0.05
            self.upper_body_joints_bounds.update({joint_index+offset : (qpos0[joint_index+offset] - distance, qpos0[joint_index+offset] + distance)})
            #print("index:", joint_index,"ind+off:",joint_index+offset, "upper_limits[index]:", upper_limits[joint_index], "lower_limits[index]:", lower_limits[joint_index], "distance", distance, "qpos0[joint_index+offset]:", qpos0[joint_index+offset])

        # for key, value in self.upper_body_joints_bounds.items():
        #     print("[DEBUG basic_locomotion_tasks]: key:", key, "value:", value)

        
        self.ideal_orientation = Quaternion(np.array([1, 0, 0, 0]))
        

    def get_reward(self):

        standing = rewards.tolerance(
            self.robot.head_height(),
            bounds=(_STAND_HEIGHT, float("inf")),
            margin=_STAND_HEIGHT / 4,
        )
        upright = rewards.tolerance(
            self.robot.torso_upright(),
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        stand_reward = standing * upright

        # I want to compute the reward for small control in the following way:
        ctrl_ranges = self.robot.get_ctrl_ranges()
        actuator_forces = np.abs(self.robot.actuator_forces()) # The ctrl range is symmetric, so I can take the absolute value.
        actuator_forces = actuator_forces/ctrl_ranges[:, 1] # I divide by the maximum value of the control range.


        control_reward = 1 - np.mean(actuator_forces**2) # I want to penalize the control signal. The reward is 1 minus the mean of the normalized control signal.
        small_control = control_reward
        #small_control = (3 + control_reward) / 4 # I want to give more importance to the other rewards than to the control_reward. It is obvious that the control signal cannot be 0.

        reward_upper_body = 0
        joint_position = self.robot.get_qpos()
        for key, (low,high) in self.upper_body_joints_bounds.items():
            #print("[DEBUG basic_locomotion_tasks]: joint_position[i]:", joint_position[key], "low:", low, "high:", high)
            reward_upper_body += rewards.tolerance(
                joint_position[key],
                bounds=(low, high),
                margin=(high - low) / 2,
                sigmoid="gaussian",
            )
        
        reward_upper_body = reward_upper_body / len(self.upper_body_joints_bounds)

        #reward_upper_body = (1 + 2*reward_upper_body) / 3


        #print("[DEBUG basic_locomotion_tasks]: self.robot.center_of_mass_velocity():", self.robot.center_of_mass_velocity())
        com_velocity_x = self.robot.center_of_mass_velocity()[0] # I take only the x component of the velocity.

        com_position_y = self.robot.center_of_mass_position()[1] # I take only the y component of the position.

        move = rewards.tolerance(
            com_velocity_x,
            bounds=(self._move_speed_lower_bound, self._move_speed_upper_bound),
            margin=self._move_speed/3,
            value_at_margin=0.1,
            sigmoid="gaussian",
        ) 
        #print("[DEBUG basic_locomotion_tasks]: self.robot.center_of_mass_velocity():", com_position_y)
        
        centered_reward = rewards.tolerance(
            com_position_y,
            bounds=(-0.3, 0.3),
            margin=0.2,
            value_at_margin=0.1,
            sigmoid="linear",
        )
        
        orientation_quat = self.robot.robot_orientation()
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


        #move = (2*move + centered_reward + stay_inline_reward)/4

        #move = (5 * move + 1) / 6
        

        reward = stand_reward*(small_control + 3*move + reward_upper_body + centered_reward + stay_inline_reward)/3


        #reward = stand_reward*(2*small_control + 5*move + 2*reward_upper_body)/9
        #print("[DEBUG basic_locomotion_tasks]: reward:", reward, "stand_reward:", stand_reward, "small_control:", small_control, "move:", move, "upper_body:", reward_upper_body)
        return reward, {
            "stand_reward": stand_reward,
            "small_control": small_control,
            "move": move,
            "standing": standing,
            "upright": upright,
        }


    # def get_reward(self):

    #     standing = rewards.tolerance(
    #         self.robot.head_height(),
    #         bounds=(_STAND_HEIGHT, float("inf")),
    #         margin=_STAND_HEIGHT / 4,
    #     )
    #     upright = rewards.tolerance(
    #         self.robot.torso_upright(),
    #         bounds=(0.9, float("inf")),
    #         sigmoid="linear",
    #         margin=1.9,
    #         value_at_margin=0,
    #     )
    #     stand_reward = standing * upright

    #     # I want to compute the reward for small control in the following way:
    #     ctrl_ranges = self.robot.get_ctrl_ranges()
    #     actuator_forces = np.abs(self.robot.actuator_forces()) # The ctrl range is symmetric, so I can take the absolute value.
    #     actuator_forces = actuator_forces/ctrl_ranges[:, 1] # I divide by the maximum value of the control range.

    #     # if np.max(actuator_forces) > 1:
    #     #     print("[DEBUG basic_locomotion_tasks] ERROR actuator_forces:", actuator_forces, "ctrl_ranges:", ctrl_ranges, "actuator_forces/ctrl_ranges[:, 1]:", actuator_forces/ctrl_ranges[:, 1])
    #     #     os.exit(0)
    #     control_reward = 1 - np.mean(actuator_forces**2) # I want to penalize the control signal. The reward is 1 minus the mean of the normalized control signal.
    #     small_control = (3 + control_reward) / 4 # I want to give more importance to the other rewards than to the control_reward. It is obvious that the control signal cannot be 0.

    #     # small_control = rewards.tolerance(
    #     #     self.robot.actuator_forces(),
    #     #     margin=10,
    #     #     value_at_margin=0,
    #     #     sigmoid="quadratic",
    #     # ).mean()
    #     # small_control = (4 + small_control) / 5

    #     if self._move_speed == 0:
    #         horizontal_velocity = self.robot.center_of_mass_velocity()[[0, 1]]
    #         dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean()
    #         return small_control * stand_reward * dont_move, {
    #             "small_control": small_control,
    #             "stand_reward": stand_reward,
    #             "dont_move": dont_move,
    #             "standing": standing,
    #             "upright": upright,
    #         }
    #     else:
    #         com_velocity = self.robot.center_of_mass_velocity()[0]
    #         move = rewards.tolerance(
    #             com_velocity,
    #             bounds=(self._move_speed, float("inf")),
    #             margin=self._move_speed,
    #             value_at_margin=0,
    #             sigmoid="linear",
    #         )
    #         move = (5 * move + 1) / 6
            
    #         reward = small_control * stand_reward * move
    #         #print("[DEBUG basic_locomotion_tasks]: reward:", reward, "stand_reward:", stand_reward, "small_control:", small_control, "move:", move, "standing:", standing, "upright:", upright)
    #         return reward, {
    #             "stand_reward": stand_reward,
    #             "small_control": small_control,
    #             "move": move,
    #             "standing": standing,
    #             "upright": upright,
    #         }


    def _print_info(self):
        """
        Useful for understanding the structure of the model. It prints information about the model.
        """
        qpos = self.robot.get_qpos()
        print("[DEBUG basic_locomotion_tasks] joint_angles:", qpos)

        home_joint_position = self.qpos0_robot["h1"]
        print("[DEBUG basic_locomotion_tasks] home position:", home_joint_position)


        print("[DEBUG basic_locomotion_tasks] dir(self.robot._env.model):", dir(self.robot._env.model))

        print("[DEBUG basic_locomotion_tasks] self.robot._env.model.name_jntadr:", self.robot._env.model.name_jntadr)
        jnt_qposadr = self.robot._env.model.jnt_qposadr
        print("[DEBUG basic_locomotion_tasks] self.robot._env.model.jnt_qposadr:", jnt_qposadr)
        print("[DEBUG basic_locomotion_tasks] self.robot._env.model.njnt:", self.robot._env.model.njnt)
        joint = self.robot._env.model.joint
        print("[DEBUG basic_locomotion_tasks] self.robot._env.model.joint:", joint)
        key_qpos = self.robot._env.model.key_qpos
        print("[DEBUG basic_locomotion_tasks] self.robot._env.model.key_qpos:", key_qpos)
        jnt_user = self.robot._env.model.jnt_user
        print("[DEBUG basic_locomotion_tasks] self.robot._env.model.jnt_user:", jnt_user)
        names = self.robot._env.model.names
        print("[DEBUG basic_locomotion_tasks] self.robot._env.model.names:", names)
        names_list = [name for name in names.split(b'\x00') if name]

        for enum, name in enumerate(names_list):
            print("[DEBUG basic_locomotion_tasks] enum:", enum, "name:", name.decode('utf-8'))

        
        name_bodyadr = self.robot._env.model.name_bodyadr
        print("[DEBUG basic_locomotion_tasks] self.robot._env.model.name_bodyadr:", name_bodyadr)

        jnt_bodyid = self.robot._env.model.jnt_bodyid
        print("[DEBUG basic_locomotion_tasks] self.robot._env.model.jnt_bodyid:", jnt_bodyid)

        jnt = self.robot._env.model.jnt
        print("[DEBUG basic_locomotion_tasks] self.robot._env.model.jnt:", jnt)

        body_jntnum = self.robot._env.model.body_jntnum
        print("[DEBUG basic_locomotion_tasks] self.robot._env.model.body_jntnum:", body_jntnum)

        body_jntadr = self.robot._env.model.body_jntadr
        print("[DEBUG basic_locomotion_tasks] self.robot._env.model.body_jntadr:", body_jntadr)

        body_treeid = self.robot._env.model.body_treeid
        print("[DEBUG basic_locomotion_tasks] self.robot._env.model.body_treeid:", body_treeid)

        body = self.robot._env.model.body
        print("[DEBUG basic_locomotion_tasks] self.robot._env.model.body:", body)

        jnt_group = self.robot._env.model.jnt_group
        print("[DEBUG basic_locomotion_tasks] self.robot._env.model.jnt_group:", jnt_group)

        jnt_limited = self.robot._env.model.jnt_limited 
        print("[DEBUG basic_locomotion_tasks] self.robot._env.model.jnt_limited:", jnt_limited)

        name_keyadr = self.robot._env.model.name_keyadr
        print("[DEBUG basic_locomotion_tasks] self.robot._env.model.name_keyadr:", name_keyadr)

        #print("[DEBUG basic_locomotion_tasks] dir(self.robot._env.model.joint):", dir(self.robot._env.model.joint))
        
        print("[DEBUG basic_locomotion_tasks] dir(self.robot._env.model.jnt):", dir(self.robot._env.model.jnt))
        print("[DEBUG basic_locomotion_tasks] dir(self.robot._env.model.body):", dir(self.robot._env.model.body))
        
        text_adr = self.robot._env.model.text_adr
        print("[DEBUG basic_locomotion_tasks] self.robot._env.model.text_adr:", text_adr)

        dof_jntid = self.robot._env.model.dof_jntid
        print("[DEBUG basic_locomotion_tasks] self.robot._env.model.dof_jntid:", dof_jntid)
        joint_name_map = self.robot._env.model.joint_name_map
        print("[DEBUG basic_locomotion_tasks] joint_name_map:", joint_name_map)
        


    def get_terminated(self):
        return self._env.data.qpos[2] < 0.2, {}

    # OVERRIDE basic task step method
    def step(self, action):

        #print("[DEBUG basic_locomotion_tasks]: action:", action)
        #action = self.unnormalize_action(action)
        #print("[DEBUG basic_locomotion_tasks]: unnormalized action:", action)

        action_high = self.robot.get_upper_limits()
        action_low = self.robot.get_lower_limits()

        desired_joint_position = (action + 1) / 2 * (action_high - action_low) + action_low
        
        action = self._env.get_joint_torques(desired_joint_position)
        
        self._env.do_simulation(action, self._env.frame_skip)

        obs = self.get_obs()
        reward, reward_info = self.get_reward()
        terminated, terminated_info = self.get_terminated()

        info = {"per_timestep_reward": reward, **reward_info, **terminated_info}
        return obs, reward, terminated, False, info
    
    def mock_next_state(self, action): # TODO: Non serve più. Da eliminare.

        data = copy.deepcopy(self._env.data)
        model = copy.deepcopy(self._env.model)


        data.ctrl[:] = action

        mujoco.mj_step(model, data, self._env.frame_skip)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(model, data)

        return np.concatenate((data.qpos.flat.copy(),data.qvel.flat.copy())) 


class Stand(Walk):
    _move_speed = 0
    success_bar = 800


class Run(Walk):
    _move_speed = _RUN_SPEED
