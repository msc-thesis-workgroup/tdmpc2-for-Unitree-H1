import os

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.spaces import Box
from dm_control.utils import rewards

# Local import
from .basic_task import Task


# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.65
_CRAWL_HEIGHT = 0.8

# Horizontal speeds above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 5


class Walk(Task):
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0",
        "h1hand": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        "h1touch": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
    }
    _move_speed = _WALK_SPEED
    htarget_low = np.array([-10.0, -2.0, 0.8])
    htarget_high = np.array([1000.0, 2.0, 2.0])
    success_bar = 700

    @property
    def observation_space(self):
        return Box(
            low=-np.inf, high=np.inf, shape=(self.robot.dof * 2 - 1,), dtype=np.float64
        )

    def get_reward(self):

        # self.robot.debug()
        #print("[DEBUG basic_locomotion_tasks]: ctrl_ranges:", ctrl_ranges)
        # print("[DEBUG basic_locomotion_tasks]: self.robot.head_height():", self.robot.head_height())
        # print("[DEBUG basic_locomotion_tasks]: self.robot.left_foot_height():", self.robot.left_foot_height())
        # print("[DEBUG basic_locomotion_tasks]: self.robot.right_foot_height():", self.robot.right_foot_height())
        # print("[DEBUG basic_locomotion_tasks]: self.robot.torso_upright():", self.robot.torso_upright())
        # print("[DEBUG basic_locomotion_tasks]: self.robot.center_of_mass_position():", self.robot.center_of_mass_position())
        # print("[DEBUG basic_locomotion_tasks]: self.robot.torso_vertical_orientation():", self.robot.torso_vertical_orientation())
        # print("[DEBUG basic_locomotion_tasks]: self.robot.joint_angles():", self.robot.joint_angles())
        # print("[DEBUG basic_locomotion_tasks]: self.robot.joint_velocities():", self.robot.joint_velocities())
        # print("[DEBUG basic_locomotion_tasks]: self.robot.control():", self.robot.control())
        # print("[DEBUG basic_locomotion_tasks]: self.robot.actuator_forces():", self.robot.actuator_forces())
        
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

        # if np.max(actuator_forces) > 1:
        #     print("[DEBUG basic_locomotion_tasks] ERROR actuator_forces:", actuator_forces, "ctrl_ranges:", ctrl_ranges, "actuator_forces/ctrl_ranges[:, 1]:", actuator_forces/ctrl_ranges[:, 1])
        #     os.exit(0)
        control_reward = 1 - np.mean(actuator_forces**2) # I want to penalize the control signal. The reward is 1 minus the mean of the normalized control signal.
        small_control = (3 + control_reward) / 4 # I want to give more importance to the other rewards than to the control_reward. It is obvious that the control signal cannot be 0.

        # small_control = rewards.tolerance(
        #     self.robot.actuator_forces(),
        #     margin=10,
        #     value_at_margin=0,
        #     sigmoid="quadratic",
        # ).mean()
        # small_control = (4 + small_control) / 5

        if self._move_speed == 0:
            horizontal_velocity = self.robot.center_of_mass_velocity()[[0, 1]]
            dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean()
            return small_control * stand_reward * dont_move, {
                "small_control": small_control,
                "stand_reward": stand_reward,
                "dont_move": dont_move,
                "standing": standing,
                "upright": upright,
            }
        else:
            com_velocity = self.robot.center_of_mass_velocity()[0]
            move = rewards.tolerance(
                com_velocity,
                bounds=(self._move_speed, float("inf")),
                margin=self._move_speed,
                value_at_margin=0,
                sigmoid="linear",
            )
            move = (5 * move + 1) / 6
            
            reward = small_control * stand_reward * move
            #print("[DEBUG basic_locomotion_tasks]: reward:", reward, "stand_reward:", stand_reward, "small_control:", small_control, "move:", move, "standing:", standing, "upright:", upright)
            return reward, {
                "stand_reward": stand_reward,
                "small_control": small_control,
                "move": move,
                "standing": standing,
                "upright": upright,
            }

    def get_terminated(self):
        return self._env.data.qpos[2] < 0.2, {}

    # OVERRIDE basic task step method
    def step(self, action):

        #print("[DEBUG basic_locomotion_tasks]: action:", action)
        #action = self.unnormalize_action(action)
        #print("[DEBUG basic_locomotion_tasks]: unnormalized action:", action)
        action_high = np.array([0.43, 0.43, 2.53, 2.05, 0.52, 0.43, 0.43, 2.53, 2.05, 0.52, 2.35, 2.87, 3.11, 4.45, 2.61, 2.87, 0.34, 1.3, 2.61])
        action_low = np.array([-0.43, -0.43, -3.14, -0.26, -0.87, -0.43, -0.43, -3.14, -0.26, -0.87, -2.35, -2.87, -0.34, -1.3,  -1.25, -2.87, -3.11, -4.45, -1.25])
        desired_joint_position = (action + 1) / 2 * (action_high - action_low) + action_low
        
        action = self._env.get_joint_torques(desired_joint_position)
        
        self._env.do_simulation(action, self._env.frame_skip)

        obs = self.get_obs()
        reward, reward_info = self.get_reward()
        terminated, terminated_info = self.get_terminated()

        info = {"per_timestep_reward": reward, **reward_info, **terminated_info}
        return obs, reward, terminated, False, info


class Stand(Walk):
    _move_speed = 0
    success_bar = 800


class Run(Walk):
    _move_speed = _RUN_SPEED
