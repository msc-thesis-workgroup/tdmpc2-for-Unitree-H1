
from .task import Task
from ..robots.robot import Robot
from ..environment import Environment

import numpy as np
from dm_control.utils import rewards
from gymnasium.spaces import Box

_STAND_HEIGHT = 1.65
_WALK_SPEED = 1

class WalkStyle1Task(Task):
    def __init__(self):
        super().__init__()
        self._observation_space = None
        self.dof = 0 # TODO: Is this necessary?

        self._move_speed = _WALK_SPEED
    

    def set_observation_space(self, robot: Robot) -> None:
        self._observation_space = Box(
            low=-np.inf, high=np.inf, shape=(robot.dof * 2 - 1,), dtype=np.float64
        )

    @property
    def observation_space(self):
        return self._observation_space

    def get_obs(self, env: Environment) -> np.array:
        position = env.data.qpos.flat.copy()
        velocity = env.data.qvel.flat.copy()
        state = np.concatenate((position, velocity))
        return state

    def unnormalize_action(self, action: np.array) -> np.array:
        # TODO: Check if this is correct
        return (
            2
            * (action - self._env.action_low)
            / (self._env.action_high - self._env.action_low)
            - 1
        )
    
    def normalize_action(self, action: np.array) -> np.array:
        # TODO: Check if this is correct
        return (action + 1) / 2 * (
            self._env.action_high - self._env.action_low
        ) + self._env.action_low
    
    def reset_model(self,env: Environment) -> np.array:
        return self.get_obs(env)
    
    def get_terminated(self, env: Environment) -> tuple[bool, dict]:
        return env.data.qpos[2] < 0.2, {}


    def get_reward(self, robot: Robot, action: np.array) -> float:
        standing = rewards.tolerance(
            robot.head_height(),
            bounds=(_STAND_HEIGHT, float("inf")),
            margin=_STAND_HEIGHT / 4,
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

        # if np.max(actuator_forces) > 1:
        #     print("[DEBUG basic_locomotion_tasks] ERROR actuator_forces:", actuator_forces, "ctrl_ranges:", ctrl_ranges, "actuator_forces/ctrl_ranges[:, 1]:", actuator_forces/ctrl_ranges[:, 1])
        #     os.exit(0)
        control_reward = 1 - np.mean(actuator_forces**2) # I want to penalize the control signal. The reward is 1 minus the mean of the normalized control signal.
        small_control = (3 + control_reward) / 4 # I want to give more importance to the other rewards than to the control_reward. It is obvious that the control signal cannot be 0.

        # small_control = rewards.tolerance(
        #     robot.actuator_forces(),
        #     margin=10,
        #     value_at_margin=0,
        #     sigmoid="quadratic",
        # ).mean()
        # small_control = (4 + small_control) / 5

        if self._move_speed == 0:
            horizontal_velocity = robot.center_of_mass_velocity()[[0, 1]]
            dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean()
            return small_control * stand_reward * dont_move, {
                "small_control": small_control,
                "stand_reward": stand_reward,
                "dont_move": dont_move,
                "standing": standing,
                "upright": upright,
            }
        else:
            com_velocity = robot.center_of_mass_velocity()[0]
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