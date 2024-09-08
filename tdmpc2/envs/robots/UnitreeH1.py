
from .robot import Robot
import numpy as np
from pyquaternion import Quaternion
import copy
from ..environment import Environment
import mujoco

import transforms3d as tf3

_DOF = 26
class H1(Robot):

    def __init__(self):
        super().__init__()
        self.dof = _DOF
    
        # TODO(my-rice): These values are hardcoded for now. We need to find a better way to get these values. Use a pd controller config file?
        #self.kp = np.array([200, 200, 200, 300, 40, 200, 200, 200, 300, 40, 300, 100, 100, 100, 100, 100, 100, 100, 100])
        # I want kp/4
        self.kp = np.array([50, 50, 50, 75, 10, 50, 50, 50, 75, 10, 75, 100, 100, 100, 100, 100, 100, 100, 100])
        
        #self.kd = np.array([5, 5, 5, 6, 2, 5, 5, 5, 6, 2, 6, 2, 2, 2, 2, 2, 2, 2, 2])
        self.kd = np.array([1.25, 1.25, 1.25, 1.5, 0.25, 1.25, 1.25, 1.25, 1.5, 0.25, 1, 2, 2, 2, 2, 2, 2, 2, 2])

        # Joint limits expressed as angular positions # TODO: Try to get these values from the environment with self._env.model.jnt_range.
        self.upper_joint_limits = np.array([0.43, 0.43, 2.53, 2.05, 0.52, 0.43, 0.43, 2.53, 2.05, 0.52, 2.35, 2.87, 3.11, 4.45, 2.61, 2.87, 0.34, 1.3, 2.61])
        self.lower_joint_limits = np.array([-0.43, -0.43, -3.14, -0.26, -0.87, -0.43, -0.43, -3.14, -0.26, -0.87, -2.35, -2.87, -0.34, -1.3,  -1.25, -2.87, -3.11, -4.45, -1.25])

        self.legs_joints = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
        self.geom_to_body_name = None

        self._env = None
        

        self.lower_body_joints = 11
        self.joints = 19

        self.lfoot_body_name = "left_ankle_link"
        self.rfoot_body_name = "right_ankle_link"
        self.robot_root_name = "pelvis"

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
    
    def _udate_robot_state_TESTING(self, model, data):
        class TestEnv:
            def __init__(self, model, data):
                self.model = model
                self.data = data
        if self._env is None:
            self._env = TestEnv(model, data)
        else:
            self._env.model = model
            self._env.data = data

    def get_nv(self):
        return self._env.model.nv
    
    def get_nq(self):
        return self._env.model.nq

    def get_time(self):
        return self._env.data.time

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
    
    def get_qacc(self):
        """Returns the joint accelerations."""
        return self._env.data.qacc.copy()

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

    def get_nv(self):
        return self._env.model.nv
    
    def get_nq(self):
        return self._env.model.nq
    
    def get_time(self):
        return self._env.data.time

    def get_qpos0(self):
        return self._env.model.key_qpos[0]

    def torso_upright(self):
        """Returns projection from z-axes of torso to the z-axes of world."""
        return self._env.named.data.xmat["torso_link", "zz"] #OK

    def torso_orientation(self):
        """Returns the orientation of the torso."""
        return self._env.named.data.xmat["torso_link"].copy()

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

    def get_lfoot_body_pos(self):
        """Returns the position of the left foot."""
        return self._env.named.data.xpos["left_ankle_link"].copy()

    def get_rfoot_body_pos(self):
        """Returns the position of the right foot."""
        return self._env.named.data.xpos["right_ankle_link"].copy()

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
        
    ### Walk V4
    def get_robot_mass(self):
        return mujoco.mj_getTotalmass(self._env.model)
    
    def get_action_dim(self):
        print("[DEBUG: robots.py]: self._env.model.nu:",self._env.model.nu)
        return self._env.model.nu

    def check_rfoot_floor_collision(self):
        """
        Returns True if there is a collision between right foot and floor.
        """
        return (len(self.get_rfoot_floor_contacts())>0)

    def get_rfoot_floor_contacts(self):
        """
        Returns list of right foot and floor contacts.
        """
        contacts = [self._env.data.contact[i] for i in range(self._env.data.ncon)]
        rcontacts = []

        rfeet = [self.rfoot_body_name] if isinstance(self.rfoot_body_name, str) else self.rfoot_body_name
        rfeet_ids = [self._env.model.body(bn).id for bn in rfeet]
        for i,c in enumerate(contacts):
            geom1_body = self._env.model.body(self._env.model.geom_bodyid[c.geom1])
            geom2_body = self._env.model.body(self._env.model.geom_bodyid[c.geom2])
            geom1_is_floor = (self._env.model.body(geom1_body.rootid).name!=self.robot_root_name)
            geom2_is_rfoot = (self._env.model.geom_bodyid[c.geom2] in rfeet_ids)
            if (geom1_is_floor and geom2_is_rfoot):
                rcontacts.append((i,c))
        return rcontacts
    
    def check_lfoot_floor_collision(self):
        """
        Returns True if there is a collision between left foot and floor.
        """
        return (len(self.get_lfoot_floor_contacts())>0)
    
    def get_lfoot_floor_contacts(self):
        """
        Returns list of left foot and floor contacts.
        """
        contacts = [self._env.data.contact[i] for i in range(self._env.data.ncon)]
        lcontacts = []

        lfeet = [self.lfoot_body_name] if isinstance(self.lfoot_body_name, str) else self.lfoot_body_name
        lfeet_ids = [self._env.model.body(bn).id for bn in lfeet]
        for i,c in enumerate(contacts):
            geom1_body = self._env.model.body(self._env.model.geom_bodyid[c.geom1])
            geom2_body = self._env.model.body(self._env.model.geom_bodyid[c.geom2])
            geom1_is_floor = (self._env.model.body(geom1_body.rootid).name!=self.robot_root_name)
            geom2_is_lfoot = (self._env.model.geom_bodyid[c.geom2] in lfeet_ids)
            if (geom1_is_floor and geom2_is_lfoot):
                lcontacts.append((i,c))
        return lcontacts
    
    def get_object_xpos_by_name(self, obj_name, object_type):
        if object_type=="OBJ_BODY":
            return self._env.data.body(obj_name).xpos
        elif object_type=="OBJ_GEOM":
            return self._env.data.geom(obj_name).xpos
        elif object_type=="OBJ_SITE":
            return self._env.data.site(obj_name).xpos
        else:
            raise Exception("object type should either be OBJ_BODY/OBJ_GEOM/OBJ_SITE.")
        
    
    def get_object_xquat_by_name(self, obj_name, object_type):
        if object_type=="OBJ_BODY":
            return self._env.data.body(obj_name).xquat
        if object_type=="OBJ_SITE":
            xmat = self._env.data.site(obj_name).xmat
            return tf3.quaternions.mat2quat(xmat)
        else:
            raise Exception("object type should be OBJ_BODY/OBJ_SITE.")
        

    def get_lfoot_body_vel(self, frame=0):
        """
        Returns translational and rotational velocity of left foot.
        """
        if isinstance(self.lfoot_body_name, list):
            return [self.get_body_vel(i, frame=frame) for i in self.lfoot_body_name]
        return self.get_body_vel(self.lfoot_body_name, frame=frame)
    
    def get_rfoot_body_vel(self, frame=0):
        """
        Returns translational and rotational velocity of right foot.
        """
        if isinstance(self.rfoot_body_name, list):
            return [self.get_body_vel(i, frame=frame) for i in self.rfoot_body_name]
        return self.get_body_vel(self.rfoot_body_name, frame=frame)
    
    def get_body_vel(self, body_name, frame=0):
        """
        Returns translational and rotational velocity of a body in body-centered frame, world/local orientation.
        """
        body_vel = np.zeros(6)
        body_id = mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        mujoco.mj_objectVelocity(self._env.model, self._env.data, mujoco.mjtObj.mjOBJ_XBODY,
                                 body_id, body_vel, frame)
        return [body_vel[3:6], body_vel[0:3]]
    

    def get_rfoot_grf(self):
        """
        Returns total Ground Reaction Force on right foot.
        """
        right_contacts = self.get_rfoot_floor_contacts()
        rfoot_grf = 0
        for i, con in right_contacts:
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self._env.model, self._env.data, i, c_array)
            rfoot_grf += np.linalg.norm(c_array)
        return rfoot_grf

    def get_lfoot_grf(self):
        """
        Returns total Ground Reaction Force on left foot.
        """
        left_contacts = self.get_lfoot_floor_contacts()
        lfoot_grf = 0
        for i, con in left_contacts:
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self._env.model, self._env.data, i, c_array)
            lfoot_grf += (np.linalg.norm(c_array))
        return lfoot_grf