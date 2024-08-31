import numpy as np

from .utility_rewards import _calc_foot_frc_clock_reward, _calc_foot_vel_clock_reward, _calc_body_orient_reward, _calc_root_accel_reward, _calc_height_reward, _calc_fwd_vel_reward, _calc_torque_reward, _calc_action_reward, create_phase_reward
from .reward import Reward

from ..robots.robot import Robot

class WalkV4(Reward):

    def __init__(self,robot: Robot):

        dt=0.01 # TODO passare a 0.01
        neutral_foot_orient=[]
        root_body='pelvis'
        lfoot_body='left_ankle_link'
        rfoot_body='right_ankle_link'
        waist_r_joint='waist_r'
        waist_p_joint='waist_p'
    

        self.robot = robot
        self._control_dt = dt
        self._neutral_foot_orient=neutral_foot_orient

        self._mass = self.robot.get_robot_mass()

        # These depend on the robot, hardcoded for now
        # Ideally, they should be arguments to __init__
        self._goal_speed_ref = 0.4 # self._goal_speed_ref = 0.4
        self._goal_height_ref = 0.98 # self._goal_height_ref = 0.98
        self._swing_duration = 0.33 # 0.75
        self._stance_duration =  0.495 # 0.35
        self._total_duration = 0.825 # 1.1


        self._root_body_name = root_body
        self._lfoot_body_name = lfoot_body
        self._rfoot_body_name = rfoot_body

        #prev_torque = np.zeros(self.robot.get_action_dim())
        self.prev_action = None 


    def get_reward(self, robot: Robot, action: np.ndarray) -> float:

        # Update the phase
        self.step()

        # Compute the reward

        self.l_foot_vel = self.robot.get_lfoot_body_vel()[0]
        self.r_foot_vel = self.robot.get_rfoot_body_vel()[0]
        self.l_foot_frc = self.robot.get_lfoot_grf()
        self.r_foot_frc = self.robot.get_rfoot_grf()
        r_frc = self.right_clock[0]
        l_frc = self.left_clock[0]
        r_vel = self.right_clock[1]
        l_vel = self.left_clock[1]
        reward_dict = dict(foot_frc_score=0.150 * _calc_foot_frc_clock_reward(self, l_frc, r_frc),
                      foot_vel_score=0.150 * _calc_foot_vel_clock_reward(self, l_vel, r_vel),
                      orient_cost=0.050 * (_calc_body_orient_reward(self, self._lfoot_body_name) +
                                           _calc_body_orient_reward(self, self._rfoot_body_name) +
                                           _calc_body_orient_reward(self, self._root_body_name))/3,
                      root_accel=0.050 * _calc_root_accel_reward(self),
                      height_error=0.050 * _calc_height_reward(self),
                      com_vel_error=0.200 * _calc_fwd_vel_reward(self),
                      #torque_penalty=0.050 * _calc_torque_reward(self, self.prev_torque),
                      #action_penalty=0.050 * _calc_action_reward(self, self.prev_action), # TODO Fa sempre 0 questo termine perchè prev_action è sempre uguale a action
                      torque_penalty=0.050 * self._calc_small_control_reward(),
                      action_penalty=0.050 * self._calc_small_action_increase(action),
        )


        #self.prev_torque = robot.actuator_forces()
        self.prev_action = action

        reward = sum(reward_dict.values())

        return reward,{}

    def step(self): # TODO
        if self._phase>self._period:
            self._phase=0
        self._phase+=1
        return

    def reset(self): # TODO
        
        self._goal_speed_ref = 0.4 #self._goal_speed_ref = np.random.choice([0, np.random.uniform(0.3, 0.4)])
        self.right_clock, self.left_clock = create_phase_reward(self._swing_duration,
                                                                        self._stance_duration,
                                                                        0.1,
                                                                        "grounded",
                                                                        1/self._control_dt)

        # number of control steps in one full cycle
        # (one full cycle includes left swing + right swing)
        self._period = np.floor(2*self._total_duration*(1/self._control_dt))
        # randomize phase during initialization
        self._phase = np.random.randint(0, self._period)

    def _calc_small_control_reward(self):
        ctrl_ranges = self.robot.get_ctrl_ranges()
        actuator_forces = np.abs(self.robot.actuator_forces())
        actuator_forces = actuator_forces/ctrl_ranges[:, 1]
        control_reward = 1 - np.mean(actuator_forces)
        return control_reward
    
    def _calc_small_action_increase(self,action):
        if self.prev_action is None:
            return 0
        
        action_increase = np.abs(action - self.prev_action)
        #action_increase = action_increase / (self.robot.action_high - self.robot.action_low)
        action_increase = np.mean(action_increase)

        return action_increase