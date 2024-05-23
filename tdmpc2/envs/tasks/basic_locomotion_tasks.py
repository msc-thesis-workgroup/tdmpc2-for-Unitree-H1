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
        small_control = rewards.tolerance(
            self.robot.actuator_forces(),
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()
        small_control = (4 + small_control) / 5
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
            return reward, {
                "stand_reward": stand_reward,
                "small_control": small_control,
                "move": move,
                "standing": standing,
                "upright": upright,
            }

    def get_terminated(self):
        return self._env.data.qpos[2] < 0.2, {}
    

    def array_to_check(self,action):
        action_high = np.array([0.43, 0.43, 2.53, 2.05, 0.52, 0.43, 0.43, 2.53, 2.05, 0.52, 2.35, 2.87, 3.11, 4.45, 2.61, 2.87, 0.34, 1.3, 2.61])
        action_low = np.array([-0.43, -0.43, -3.14, -0.26, -0.87, -0.43, -0.43, -3.14, -0.26, -0.87, -2.35, -2.87, -0.34, -1.3,  -1.25, -2.87, -3.11, -4.45, -1.25])

        #TODO put action_high and action_low to replace self._env.action_high and self._env.action_low
        desired_joint_position = (action + 1) / 2 * (
            self._env.action_high - self._env.action_low
        ) + self._env.action_low    


        #desired_joint_position = (action + 1) / 2 * (action_high - action_low) + action_low
        
        

        torques = self._env.get_joint_torques(desired_joint_position)

        torque_limits = np.array([200,200,200,300,40,200,200,200,300,40,200,40,40,18,18,40,40,18,18])

        # check if the torques are within the limits
        for id,value in enumerate(torques):
            if value > torque_limits[id]:
                torques[id] = torque_limits[id]
            elif value < -torque_limits[id]:
                torques[id] = -torque_limits[id]

        self._env.data.qfrc_actuator[6:self.robot.dof] = torques
        #print("check: ", check)
        #return check



    def step(self, action):
        
        old_action = action
        #check = self.array_to_check(action)
        
        action = self.unnormalize_action(action)
        #print("action: ", action)

        # if not np.array_equal(np.round(check,3), np.round(action,3)):
        #     print("action: ", action)
        #     print("check: ", check)

        # qfrc_inverse = self._env.data.qfrc_inverse
        # print("qfrc_inverse: ", qfrc_inverse)

        # qfrc_bias = self._env.data.qfrc_bias
        # print("qfrc_bias: ", qfrc_bias)

        # qfrc_smooth = self._env.data.qfrc_smooth
        # print("qfrc_smooth: ", qfrc_smooth)

        # qfrc_spring = self._env.data.qfrc_spring
        # print("qfrc_spring: ", qfrc_spring)

        # Get the joint torques from mujoco environment
        #applied = self._env.data.qfrc_applied[:self.robot.dof] 
        actuator = self._env.data.qfrc_actuator[6:self.robot.dof] # == actuator_force
        passive = self._env.data.qfrc_passive[:self.robot.dof]
        
        #actuator_force = self._env.data.actuator_force[:self.robot.dof]
        #print("BEFORE ctrl: ", actuator_force)
        

        #print("applied: ", applied)
        #print("actuator: ", actuator)
        #print("check - actuator: ", check - actuator)

        #print("passive: ", passive)

        #print("dir(env): ", dir(self._env))
        self.array_to_check(old_action)

        self._env.do_simulation(self._env.data.ctrl, self._env.frame_skip)
        
        # Get the joint torques from mujoco environment
        #applied = self._env.data.qfrc_applied[:self.robot.dof] 
        actuator = self._env.data.qfrc_actuator[6:self.robot.dof] # == actuator_force
        
        #check = self.array_to_check(old_action)
        
        #print("AFTER check: ", check)
        #print("AFTER actuator: ", actuator)
        #print("AFTER check - actuator: ", check - actuator)



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
