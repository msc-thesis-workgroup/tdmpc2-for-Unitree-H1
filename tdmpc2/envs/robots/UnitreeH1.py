
from .robot import Robot
import numpy as np
from pyquaternion import Quaternion
import copy
from ..environment import Environment
import mujoco


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

        self.legs_joints = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
        self.geom_to_body_name = None

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
    
    def get_feet_contacts(self):
        """Returns the contact forces of the feet."""
        model = self._env.model
        data = self._env.data
        # Create a dictionary with geom_id as key and body_name as value
        
        if self.geom_to_body_name is None:
            self.geom_to_body_name = {}
            for geom_id in range(model.ngeom):
                body_name = self._get_body_name_from_geom_id(model, geom_id)
                self.geom_to_body_name[geom_id] = body_name

            # Print the dictionary
            # for geom_id, body_name in self.geom_to_body_name.items():
            #     print(f'Geom ID: {geom_id}, Body Name: {body_name}')

            self.body_name_to_geom_ids = {}
            for geom_id, body_name in self.geom_to_body_name.items():
                if body_name not in self.body_name_to_geom_ids:
                    self.body_name_to_geom_ids[body_name] = []
                self.body_name_to_geom_ids[body_name].append(geom_id)

            # print("\nBody Name to Geom IDs Dictionary:")
            # for body_name, geom_ids in self.body_name_to_geom_ids.items():
            #     print(f'Body Name: {body_name}, Geom IDs: {geom_ids}')

        left_ankle_link_geom_ids = self.body_name_to_geom_ids['left_ankle_link']
        right_ankle_link_geom_ids = self.body_name_to_geom_ids['right_ankle_link']
        #print("[DEBUG: robots.py]: left_ankle_link_geom_ids:",left_ankle_link_geom_ids,"right_ankle_link_geom_ids:",right_ankle_link_geom_ids)
        left_foot_contact = self._get_norm_contacts(model = model,data=data, geom_ids=left_ankle_link_geom_ids)
        right_foot_contact = self._get_norm_contacts(model = model,data=data, geom_ids=right_ankle_link_geom_ids)
        #print("[DEBUG: robots.py]: left_foot_contact:",left_foot_contact,"right_foot_contact:",right_foot_contact)
        return left_foot_contact, right_foot_contact
    
    def get_qpos(self):
        """Returns the joint positions."""
        return self._env.data.qpos.copy()

    def get_qvel(self):
        """Returns the joint velocities."""
        return self._env.data.qvel.copy()
    

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
        
        quat = data_temp.xquat["torso_link"].copy()
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


        quat = self._env.named.data.xquat["left_ankle_link"].copy()
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

        quat = self._env.named.data.xquat["right_ankle_link"].copy()
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
    

    def normalized_actuator_forces_legs_joint(self):
        """Returns the normalized forces applied by the actuators in the legs joint."""
        actuator_forces = list()
        ctrl_ranges = list()
        for name in self.legs_joints:
            actuator_forces.append(self._env.named.data.actuator_force[name])
            ctrl_ranges.append(self._env.named.model.actuator_ctrlrange[name])
        
            #print("[DEBUG basic_locomotion_tasks]: name:", name, "actuator_force:", actuator_forces[-1],"ctrl_range:", ctrl_ranges[-1])
        
        
        actuator_forces = np.abs(np.array(actuator_forces)) # The ctrl range is symmetric, so I can take the absolute value.
        ctrl_ranges = np.array(ctrl_ranges)
        actuator_forces = actuator_forces/ctrl_ranges[:, 1] # I divide by the maximum value of the control range to normalize the values.
        return actuator_forces
    
    def _get_body_name_from_geom_id(self,model, geom_id):
        body_id = model.geom_bodyid[geom_id]
        name_start = model.name_bodyadr[body_id]
        name_end = model.names.find(b'\0', name_start)
        return model.names[name_start:name_end].decode('utf-8')
    
    def _get_norm_contacts(self, model,data, geom_ids):
        value = 0
        for i in range(data.ncon): # number of detected contacts
            # Note that the contact array has more than `ncon` entries, so be careful to only read the valid entries.
            contact = data.contact[i]
            if contact.geom2 in geom_ids: 
            
                # print('contact', i)
                # print('dist', contact.dist)
                # print("geom", contact.geom) # array that contains the geom ids of the two bodies in contact (geom1, geom2)
                # print('geom1', contact.geom1) # world
                # print('geom2', contact.geom2) # geom del corpo in contatto
                # geom1 = contact.geom1
                # geom2 = contact.geom2
                # print("contact between bodies:", self.geom_to_body_name[geom1], "and ", self.geom_to_body_name[geom2])
                
                # Use internal functions to read out mj_contactForce
                c_array = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(model, data, i, c_array) # Extract 6D force:torque given contact id, in the contact frame.
                # A 6D vector specifying the collision forces/torques[3D force + 3D torque]
                
                #print('mj_contactForce c_array:', c_array)
                #print('norm real', np.sqrt(np.sum(np.square(c_array))))

                value += np.sqrt(np.sum(np.square(c_array)))
        return value
               

    def get_body_pos(self,body_name):

        if body_name == "left_foot":
            body_name = "left_ankle_link"
        elif body_name == "right_foot":
            body_name = "right_ankle_link"
        elif body_name == "left_knee":
            body_name = "left_knee_link"
        elif body_name == "right_knee":
            body_name = "right_knee_link"
        
        #print("[DEBUG: robots.py]: body_name:",body_name)
        return self._env.named.data.xpos[body_name].copy()
    def get_subtree_linvel(self,body_name):
        
        return self._env.named.data.subtree_linvel[body_name].copy()
        