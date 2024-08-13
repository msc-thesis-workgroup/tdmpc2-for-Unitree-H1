
from .robot import Robot
import numpy as np
from pyquaternion import Quaternion
import copy
from ..environment import Environment


_DOF = 26
class H1(Robot):

    def __init__(self):
        super().__init__()
        self.dof = _DOF
    
        # TODO(my-rice): These values are hardcoded for now. We need to find a better way to get these values. Use a pd controller config file?
        self.kp = np.array([200, 200, 200, 300, 40, 200, 200, 200, 300, 40, 300, 100, 100, 100, 100, 100, 100, 100, 100])
        self.kd = np.array([5, 5, 5, 6, 2, 5, 5, 5, 6, 2, 6, 2, 2, 2, 2, 2, 2, 2, 2])

        # Joint limits expressed as angular positions # TODO: Try to get these values from the environment with self._env.model.jnt_range.
        self.upper_joint_limits = np.array([0.43, 0.43, 2.53, 2.05, 0.52, 0.43, 0.43, 2.53, 2.05, 0.52, 2.35, 2.87, 3.11, 4.45, 2.61, 2.87, 0.34, 1.3, 2.61])
        self.lower_joint_limits = np.array([-0.43, -0.43, -3.14, -0.26, -0.87, -0.43, -0.43, -3.14, -0.26, -0.87, -2.35, -2.87, -0.34, -1.3,  -1.25, -2.87, -3.11, -4.45, -1.25])


        # self._env.model.jnt_range:          [[ 0.    0.  ]
        #                                     [-0.43  0.43]
        #                                     [-0.43  0.43]
        #                                     [-1.57  1.57]
        #                                     [-0.26  2.05]
        #                                     [-0.87  0.52]
        #                                     [-0.43  0.43]
        #                                     [-0.43  0.43]
        #                                     [-1.57  1.57]
        #                                     [-0.26  2.05]
        #                                     [-0.87  0.52]
        #                                     [-2.35  2.35]
        #                                     [-2.87  2.87]
        #                                     [-0.34  3.11]
        #                                     [-1.3   4.45]
        #                                     [-1.25  2.61]
        #                                     [-2.87  2.87]
        #                                     [-3.11  0.34]
        #                                     [-4.45  1.3 ]
        #                                     [-1.25  2.61]]

    # Override the update_robot_state method from the Robot class.
    def update_robot_state(self, env: Environment):
        """Updates the robot state with the environment data."""
        self._env = env # TODO: replace this so it only gets the necessary data from the environment.
    
    def get_qpos(self):
        """Returns the joint positions."""
        return self._env.data.qpos.copy()

    def get_upper_limits(self):
        """Returns the upper limits of the joints. These are the maximum angular positions that the joints can reach."""
        return self.upper_joint_limits

    def get_lower_limits(self):
        """Returns the lower limits of the joints. These are the minimum angular positions that the joints can reach."""
        return self.lower_joint_limits

    def get_kp(self):
        """Returns the proportional gains for the joints."""
        return self.kp
    
    def get_kd(self):
        """Returns the derivative gains for the joints."""
        return self.kd

    def torso_upright(self):
        """Returns projection from z-axes of torso to the z-axes of world."""
        return self._env.named.data.xmat["torso_link", "zz"] #OK

    def head_height(self):
        """Returns the height of the torso."""
        # Get torso_link position as a body in the space and add "0 0 0.7" as an offset
        #data_temp = copy.deepcopy(self._env.named.data)
        data_temp = self._env.named.data
        
        quat = data_temp.xquat["torso_link"]
        offset = np.array([0, 0, 0.7])
        offset = Quaternion(quat).rotate(offset)
        head_pos = data_temp.xpos["torso_link"] + offset #np.array([0, 0, 0.7])

        #temp = data_temp.site_xpos["head"]
        #print("[DEBUG: robots.py] calculated head_pos:",head_pos,"self._env.named.data.xpos['head']",temp,"EQUAL?",head_pos==temp)
        return head_pos[2]
        #return self._env.named.data.site_xpos["head", "z"] # Problem here

    def left_foot_height(self):
        """Returns the height of the left foot."""
        #data_temp = copy.deepcopy(self._env.named.data)


        quat = self._env.named.data.xquat["left_ankle_link"]
        offset = np.array([0, 0, -0.05])
        offset = Quaternion(quat).rotate(offset)
        
        #left_foot_height_pos = self._env.named.data.subtree_com["left_ankle_link"].copy() + offset
        left_foot_height_pos = self._env.named.data.xpos["left_ankle_link"].copy() + offset
        #temp = self._env.named.data.site_xpos["left_foot"]
        #print("[DEBUG: robots.py]: left_foot_height_pos:",left_foot_height_pos, "self._env.named.data.site_xpos['left_foot']: ",temp, "EQUAL?",left_foot_height_pos==temp) 
        return left_foot_height_pos[2]
        #return self._env.named.data.site_xpos["left_foot", "z"] # Problem here

    def right_foot_height(self):
        """Returns the height of the right foot."""

        quat = self._env.named.data.xquat["right_ankle_link"]
        offset = np.array([0, 0, -0.05])
        offset = Quaternion(quat).rotate(offset)

        #right_foot_height_pos = self._env.named.data.subtree_com["right_ankle_link"].copy() + offset
        right_foot_height_pos = self._env.named.data.xpos["right_ankle_link"].copy() + offset

        #temp = self._env.named.data.site_xpos["right_foot"]
        #print("[DEBUG: robots.py]: right_foot_height_pos:",right_foot_height_pos, "self._env.named.data.site_xpos['right_foot']: ",temp, "EQUAL?",right_foot_height_pos==temp)

        return right_foot_height_pos[2]
        #return self._env.named.data.site_xpos["right_foot", "z"] # Problem here

    def center_of_mass_position(self):
        """Returns position of the center-of-mass."""
        return self._env.named.data.subtree_com["pelvis"].copy() # OK

    def center_of_mass_velocity(self):
        """Returns the velocity of the center-of-mass."""
        #print("[DEBUG: robots.py]: center_of_mass_velocity:",self._env.named.data.sensordata["pelvis_subtreelinvel"].copy())
        return self._env.named.data.sensordata["pelvis_subtreelinvel"].copy() # Problem here

    def robot_velocity(self):
        """Returns the velocity of the robot in the global frame."""
        return self._env.named.data.qvel[0:3].copy()
    
    def robot_position(self):
        """Returns the position of the robot in the global frame."""
        return self._env.named.data.qpos[0:3].copy()

    def body_velocity(self):
        """Returns the velocity of the torso in local frame."""
        # Get the velocity of the pelvis
        return self._env.named.data.sensordata["body_velocimeter"].copy() # Problem here

    def torso_vertical_orientation(self):
        """Returns the z-projection of the torso orientation matrix."""
        return self._env.named.data.xmat["torso_link", ["zx", "zy", "zz"]] # OK

    def joint_angles(self):
        """Returns the state without global orientation or position."""
        # Skip the 7 DoFs of the free root joint.
        return self._env.data.qpos[7 : self.dof].copy() # OK

    def joint_velocities(self):
        """Returns the joint velocities."""
        return self._env.data.qvel[6 : self.dof].copy() # OK
    
    def robot_orientation(self):
        """Returns the orientation of the robot."""
        return self._env.data.qpos[3:7].copy()

    def control(self):
        """Returns a copy of the control signals for the actuators."""
        return self._env.data.ctrl.copy()
    
    def get_ctrl_ranges(self):
        """Returns a copy of the control ranges for the actuators."""
        return self._env.model.actuator_ctrlrange.copy()

    def actuator_forces(self):
        """Returns a copy of the forces applied by the actuators."""
        return self._env.data.actuator_force.copy()
