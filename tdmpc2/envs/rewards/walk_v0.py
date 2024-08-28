
import numpy as np
from .reward import Reward
from dm_control.utils import rewards
from ..robots.robot import Robot

class WalkV0(Reward):

    def __init__(self, robot: Robot = None):
        super().__init__()
        self._stand_height = 1.65
        self._walk_speed = 1

    def set_stand_height(self, stand_height):
        self._stand_height = stand_height

    def reset(self) -> None:
        pass

    def set_walk_speed(self, walk_speed):
        self._walk_speed = walk_speed

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