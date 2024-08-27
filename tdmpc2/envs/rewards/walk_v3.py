
import numpy as np
from .reward import Reward
from dm_control.utils import rewards
from ..robots.robot import Robot
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion


class Parameters:
    base_height_target = 0.90
    min_dist = 0.2
    max_dist = 0.5
    # put some settings here for LLM parameter tuning
    target_joint_pos_scale = 0.17    # rad
    target_feet_height = 0.06       # m
    cycle_time = 0.64                # sec
    # if true negative total rewards are clipped at zero (avoids early termination problems)
    only_positive_rewards = True
    # tracking reward = exp(error*sigma)
    tracking_sigma = 5
    max_contact_force = 1000  # Forces above this value are penalized

    class scales:
        # reference motion tracking
        joint_pos = 1.6
        feet_clearance = 1.
        feet_contact_number = 1.2
        # gait
        feet_air_time = 1.
        foot_slip = -0.05
        feet_distance = 0.2
        knee_distance = 0.2
        # contact
        feet_contact_forces = -0.01
        # vel tracking
        tracking_lin_vel = 1.0
        tracking_ang_vel = 0.5
        vel_mismatch_exp = 0.5  # lin_z; ang x,y
        low_speed = 0.2
        track_vel_hard = 0.5
        # base pos
        default_joint_pos = 0.5
        orientation = 1.0
        base_height = 0.2
        base_acc = 0.2
        # energy
        action_smoothness = -0.002
        torques = -1e-5
        dof_vel = -5e-4
        dof_acc = -1e-7
        collision = -1.

class WalkV3(Reward):

    def __init__(self, robot: Robot):
        super().__init__()
        self.parameters = Parameters()
        self.num_envs = 4096 # Probabilmente sono il numero di environment in contemporanea.
        self.dt = 0.002 #TODO
        self.dof_pos = robot.get_qpos()

        self.feet_air_time = np.zeros(2)

        lin_vel_x = [-0.3, 0.6]  # min max [m/s]
        lin_vel_y = [-0.3, 0.3]   # min max [m/s]
        ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]
        heading = [-3.14, 3.14]
        
        self.commands = np.array([lin_vel_x, lin_vel_y, ang_vel_yaw, heading])

        self.dof_vel = np.zeros(19)
        self.last_dof_vel = np.zeros_like(self.dof_pos)

        self.last_actions = np.zeros(19)
        self.last_last_actions = np.zeros(19)

        

    def get_reward(self, robot: Robot, action: np.ndarray|list[np.ndarray]) -> float:
        self.robot = robot
        self.actions = action
        self.compute_ref_state()
        self.qpos = robot.get_qpos()

        self.dof_pos = self.qpos[7:26] # TODO Separate the joint positions from the root state. In addition the dof_pos here is only the legs.
        self.dof_vel = robot.get_qvel()[6:25] 
        self.root_states = self.qpos[0:7]

        self.base_quat = self.qpos[3:7] # self.base_quat[:] = self.root_states[:, 3:7]
        quat = Quaternion(self.base_quat)
        #self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_lin_vel = quat.inverse.rotate(self.qpos[26:29]) # 26:29 is the linear velocity of the base
        #self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.base_ang_vel = quat.inverse.rotate(self.qpos[29:32]) # 29:32 is the angular velocity of the base
        
        left_contact, right_contact = self.robot.get_feet_contacts()
        self.contact_forces = np.stack([left_contact, right_contact], axis=1)

        self.torques = self.robot.actuator_forces()
        ### 

        reward = self.compute_reward()

        self.last_last_actions = self.last_actions
        self.last_actions = self.action
        self.last_dof_vel = self.dof_vel


        return reward
    
    def compute_reward(self):
        reward = 0
        reward += self.parameters.scales.joint_pos * self._reward_joint_pos()
        reward += self.parameters.scales.feet_distance * self._reward_feet_distance()
        reward += self.parameters.scales.knee_distance * self._reward_knee_distance()
        reward += self.parameters.scales.foot_slip * self._reward_foot_slip()
        reward += self.parameters.scales.feet_air_time * self._reward_feet_air_time()
        reward += self.parameters.scales.feet_contact_number * self._reward_feet_contact_number()
        reward += self.parameters.scales.feet_contact_forces * self._reward_feet_contact_forces()
        reward += self.parameters.scales.default_joint_pos * self._reward_default_joint_pos()
        reward += self.parameters.scales.base_height * self._reward_base_height()
        reward += self.parameters.scales.base_acc * self._reward_base_acc()
        reward += self.parameters.scales.vel_mismatch_exp * self._reward_vel_mismatch_exp()
        reward += self.parameters.scales.tracking_lin_vel * self._reward_tracking_lin_vel()
        reward += self.parameters.scales.tracking_ang_vel * self._reward_tracking_ang_vel()
        reward += self.parameters.scales.low_speed * self._reward_low_speed()
        reward += self.parameters.scales.track_vel_hard * self._reward_track_vel_hard()
        reward += self.parameters.scales.action_smoothness * self._reward_action_smoothness()
        reward += self.parameters.scales.torques * self._reward_torques()
        reward += self.parameters.scales.dof_vel * self._reward_dof_vel()
        # reward += self.parameters.scales.collision * self._reward_collision()
        reward += self.parameters.scales.dof_acc * self._reward_dof_acc()
        return reward

    def  _get_phase(self):
        cycle_time = self.parameters.cycle_time
        # self.episode_length_buf are the number of iterations in the episode. This * dt gives the time in seconds.
        phase = self.episode_length_buf * self.dt / cycle_time # TODO Fill episode_length_buf
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = np.sin(2 * np.pi * phase)
        # Add double support phase  
        
        stance_mask = np.zeros(2)
        # left foot stance
        stance_mask[0] = sin_pos >= 0
        # right foot stance
        stance_mask[1] = sin_pos < 0
        # Double support phase
        stance_mask[np.abs(sin_pos) < 0.1] = 1

        return stance_mask
    
    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = np.sin(2 * np.pi * phase)
        sin_pos_l = sin_pos.copy()
        sin_pos_r = sin_pos.copy()
        self.ref_dof_pos = np.zeros_like(self.dof_pos)
        scale_1 = self.parameters.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        sin_pos_l[np.abs(sin_pos_l) < 0.1] = 0
        # TODO robot.get_home_joint_pos('left_hip_pitch_joint') etc ...
        # TODO Refactor this. ref_dof_pos will become the joint reference position. In particular for legs. 
        self.ref_dof_pos[:, 2] =  sin_pos_l * scale_1 + self.cfg.init_state.default_joint_angles['left_hip_pitch_joint']
        self.ref_dof_pos[:, 3] =  sin_pos_l * scale_2 + self.cfg.init_state.default_joint_angles['left_knee_joint']
        self.ref_dof_pos[:, 4] =  sin_pos_l * scale_1 + self.cfg.init_state.default_joint_angles['left_ankle_joint']
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        sin_pos_r[np.abs(sin_pos_r) < 0.1] = 0
        self.ref_dof_pos[:, 7] = sin_pos_r * scale_1 - self.cfg.init_state.default_joint_angles['right_hip_pitch_joint']
        self.ref_dof_pos[:, 8] = sin_pos_r * scale_2 - self.cfg.init_state.default_joint_angles['right_knee_joint']
        self.ref_dof_pos[:, 9] = sin_pos_r * scale_1 - self.cfg.init_state.default_joint_angles['right_ankle_joint']
        # # Double support phase
        # self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

        self.ref_action = 2 * self.ref_dof_pos

    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        """
        joint_pos = self.robot_position()[7:26] # TODO Check this
        pos_target = self.ref_dof_pos.copy()
        diff = joint_pos - pos_target
        r = np.exp(-2 * np.linalg.norm(diff, axis=1)) - 0.2 * np.clip(np.linalg.norm(diff, axis=1),0, 0.5)
        return r

    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2] # TODO
        foot_dist = np.linalg.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], axis=1)
        fd = self.parameters.min_dist
        max_df = self.parameters.max_dist
        d_min = np.clip(foot_dist - fd, -0.5, 0.)
        d_max = np.clip(foot_dist - max_df, 0, 0.5)
        return (np.exp(-np.abs(d_min) * 100) + np.exp(-np.abs(d_max) * 100)) / 2

    def _reward_knee_distance(self):
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        """
        foot_pos = self.rigid_state[:, self.knee_indices, :2] # TODO Get the knee indices
        foot_dist = np.linalg.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], axis=1)
        fd = self.parameters.min_dist
        max_df = self.parameters.max_dist / 2

        d_min = np.clip(foot_dist - fd, -0.5, 0.)
        d_max = np.clip(foot_dist - max_df, 0, 0.5)
        return (np.exp(-np.abs(d_min) * 100) + np.exp(-np.abs(d_max) * 100)) / 2

    def _reward_foot_slip(self):
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        """
        #contact = self.contact_forces[:, self.feet_indices, 2] > 5. # TODO mjData.contact
        contact = self.contact_forces > 5
        foot_speed_norm = np.linalg.norm(self.rigid_state[:, self.feet_indices, 10:12], axis=2) # TODO
        rew = np.sqrt(foot_speed_norm)
        rew *= contact
        return np.sum(rew, axis=1)   

    def _reward_feet_air_time(self):
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        """
        #contact = self.contact_forces[:, self.feet_indices, 2] > 5. # TODO mjData.contact
        contact = self.contact_forces > 5
        
        stance_mask = self._get_gait_phase()
        self.contact_filt = np.logical_or(np.logical_or(contact, stance_mask), self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        self.feet_air_time *= ~self.contact_filt
        return air_time.sum(dim=1)

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        """
        #contact = self.contact_forces[:, self.feet_indices, 2] > 5. # TODO mjData.contact
        contact = self.contact_forces > 5
        
        stance_mask = self._get_gait_phase()
        reward = np.where(contact == stance_mask, 1, -0.3)
        return np.mean(reward, axis=1)

    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation 
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        #
        self.base_quat[:] = self._env.data.qpos[3:7].copy() # TODO define base quat
        self.base_euler_xyz = R.from_quat(self.base_quat,scalar_first=True).as_euler('xyz', degrees=False)
        # 
        # Rotate a vector by the inverse of a quaternion
        # self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.gravity_vec = np.array([0, 0, -9.81])
        quat = Quaternion(self.base_quat)
        self.projected_gravity = quat.inverse.rotate(self.gravity_vec)
        #
        quat_mismatch = np.exp(-np.sum(np.abs(self.base_euler_xyz[:2]), axis=1) * 10)
        orientation = np.exp(-np.norm(self.projected_gravity[:, :2], axis=1) * 20)
        return (quat_mismatch + orientation) / 2.

    def _reward_feet_contact_forces(self):
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        """
        #return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 400), dim=1)

        # contact_forces = self.contact_forces[:, self.feet_indices, :]
        contact = self.contact_forces
        
        max_contact_force = self.parameters.max_contact_force

        # Compute the norm along the last dimension
        norms = np.abs(contact) # It is already the norm
        
        # Subtract max_contact_force and clip the values between 0 and 400
        clipped_norms = np.clip(norms - max_contact_force, 0, 400)

        # Sum along the second dimension
        result = np.sum(clipped_norms)

        return result
        
    def _reward_default_joint_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """

        # TODO: Subtract to qpos the default joint angles. Then exclude left_hip_yaw_joint, left_hip_roll_joint and right_hip_yaw_joint, right_hip_roll_joint
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.4,         
           'left_knee_joint' : 0.8,       
           'left_ankle_joint' : -0.4,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.4,                                       
           'right_knee_joint' : 0.8,                                             
           'right_ankle_joint' : -0.4                                     
        }

        default_joint_pd_target = default_joint_angles.values()
        joint_diff = self.dof_pos - default_joint_pd_target

        # TODO

        left_yaw_roll = joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 6: 8]
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_base_height(self):
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        """
        stance_mask = self._get_gait_phase()

        # TODO What is the root state? root state is pos = [0.0, 0.0, 1.] # x,y,z [m] rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        # In my case it is [0 0 0.98 1 0 0 0] -> qpos[0:7]
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)
        return torch.exp(-torch.abs(base_height - self.parameters.base_height_target) * 100)

    def _reward_base_acc(self):
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        """
        # TODO
        root_acc = self.last_root_vel - self.root_states[:, 7:13] # Why doesn't divide by dt?
        
        #rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        root_acc = np.array(root_acc)
        norms = np.linalg.norm(root_acc)
        rew = np.exp(-norms * 3)
        return rew


    def _reward_vel_mismatch_exp(self):
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        """

        # Compute lin_mismatch
        lin_mismatch = np.exp(-np.square(self.base_lin_vel[2]) * 10)

        # Compute ang_mismatch
        ang_mismatch = np.exp(-np.linalg.norm(self.base_ang_vel[2]) * 5)

        # Compute c_update
        c_update = (lin_mismatch + ang_mismatch) / 2

        return c_update

    def _reward_track_vel_hard(self):
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        """

        # Compute linear velocity error
        lin_vel_error = np.linalg.norm(self.commands[:2] - self.base_lin_vel[:2])
        lin_vel_error_exp = np.exp(-lin_vel_error * 10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = np.abs(self.commands[2] - self.base_ang_vel[2])
        ang_vel_error_exp = np.exp(-ang_vel_error * 10)

        # Compute linear error
        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        # Compute the final result
        result = (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

        return result

    def _reward_tracking_lin_vel(self):
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        """
        # Compute lin_vel_error
        lin_vel_error = np.sum(np.square(self.commands[:2] - self.base_lin_vel[:2]), axis=1)

        # Compute the final result
        result = np.exp(-lin_vel_error * self.parameters.tracking_sigma)

        return result

    def _reward_tracking_ang_vel(self):
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        """   
        # Compute ang_vel_error
        ang_vel_error = np.square(self.commands[2] - self.base_ang_vel[2])

        # Compute the final result
        result = np.exp(-ang_vel_error * self.parameters.tracking_sigma)

        return result

    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        """
        # Compute feet contact mask
        contact = self.contact_forces > 5

        # Get the z-position of the feet and compute the change in z-position
        feet_z = self.rigid_state[self.feet_indices, 2] - 0.05 # TODO
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z # TODO
        self.last_feet_z = feet_z

        # Compute swing mask
        swing_mask = 1 - self._get_gait_phase()

        # feet height should be closed to target feet height at the peak
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
        self.feet_height *= ~contact
        return rew_pos

    def _reward_low_speed(self):
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        """
        # Compute absolute values
        absolute_speed = np.abs(self.base_lin_vel[0])
        absolute_command = np.abs(self.commands[0])

        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # Check if the speed and command directions are mismatched
        sign_mismatch = np.sign(self.base_lin_vel[0]) != np.sign(self.commands[0])

        # Initialize reward array
        reward = np.zeros_like(self.base_lin_vel[0])

        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.0
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0

        # Return the reward multiplied by the condition
        return reward * (np.abs(self.commands[0]) > 0.1)

    def _reward_torques(self):
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        """
        return np.sum(np.square(self.torques))

    def _reward_dof_vel(self):
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and 
        more controlled movements.
        """
        return np.sum(np.square(self.dof_vel))

    def _reward_dof_acc(self):
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        """
        return np.sum(np.square((self.last_dof_vel - self.dof_vel) / self.dt))

    # def _reward_collision(self):
    #     """
    #     Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
    #     This encourages the robot to avoid undesired contact with objects or surfaces.
    #     """
    #     # TODO Remove this function. It doesn't make sense to penalize the robot for collisions if there are no objects in the environment.
    #     return np.sum(1.*(np.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = np.sum(np.square(
            self.last_actions - self.actions))
        term_2 = np.sum(np.square(
            self.actions + self.last_last_actions - 2 * self.last_actions))
        term_3 = 0.05 * np.sum(np.abs(self.actions))
        return term_1 + term_2 + term_3
