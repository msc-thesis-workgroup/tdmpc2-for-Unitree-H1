
import numpy as np
from .reward import Reward
from dm_control.utils import rewards
from ..robots.robot import Robot
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.65 # The ideal height of the head is 1.68 m.

class HybridWalkBENCHMARK(Reward):

    def __init__(self,robot: Robot):
        super().__init__()
        print("[DEBUG basic_locomotion_tasks]: WalkV0Easy.__init__()")
        self._stand_height = _STAND_HEIGHT
        self._move_speed_lower_bound = 0.55 #0.83 # 3 km/h # 2 km/h = 0.55 
        self._move_speed_upper_bound = 1.5 # 6.4 km/h #4 km/h = 1.11 m/s

        self.reset()
        self.robot = robot

        self.goal = np.array([9, 9])
        self.home = np.array([1.5, 0])
        self.max_distance = np.linalg.norm(self.goal - self.home)
        self.prev_dist = self.max_distance

    def reset(self) -> None:
        pass

    def get_reward(self, robot: Robot, action: np.ndarray) -> float:
        self.robot = robot
        standing = rewards.tolerance(
            robot.head_height(),
            bounds=(self._stand_height, float("inf")),
            margin=self._stand_height/6,
            value_at_margin=0,
            sigmoid="linear",
        )
        upright = rewards.tolerance(
            robot.torso_upright(),
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9, # it doesn't make sense
            value_at_margin=0,
        )
        stand_reward = standing * upright

        ctrl_ranges = robot.get_ctrl_ranges()[0:11]
        actuator_forces = np.abs(robot.actuator_forces()[0:11]) # The ctrl range is symmetric, so I can take the absolute value.
        # Get actuator forces only for the lower body joints
        actuator_forces = actuator_forces/ctrl_ranges[:, 1] # I divide by the maximum value of the control range.
        actuator_forces = np.mean(actuator_forces)
        control_reward = 1 - actuator_forces

        #velocity_x = robot.center_of_mass_velocity()[0] # robot.robot_velocity()[0] # I take only the x component of the velocity.
        #print("velocity_x",velocity_x)
        position = robot.robot_position() # I take only the y component of the position.
        #position_y = position[1]

        position = position[0:2]
        distance = np.linalg.norm(self.goal - position)
        # if distance < 0 or distance > self.max_distance:
        #     move = 0
        # else:
        #     move = distance / self.max_distance
        #     move = 1 - move

        eps = 0.001
        if (distance) <= self.prev_dist:
            move = 1
        else:
            move = 0
        
        move = (1+2*move)/3

        # avoid_obstacles = 0
        # if position[0] >= 3.5 and position[1] <= 2:
        #     print("Collinding with obstacle: -50")
        #     avoid_obstacles = -50
        # elif (position[0] >= 2.7 and position[0] <= 7.3) and (position[1] >= 3 and position[1] <= 6):
        #     print("Collinding with obstacle: -50")
        #     avoid_obstacles = -50

        flag = check_collision(robot._env)
        if flag:
            avoid_obstacles = -50
        else:
            avoid_obstacles = 0

        # fix torso orientation
        # torso_jnt = robot.get_qpos()[17]
        
        # stay_inline_reward = rewards.tolerance(
        #     torso_jnt,
        #     bounds=(-0.35, 0.35), # -20 degrees, 20 degrees
        #     margin=0.45, # 25 degrees # (-
        #     value_at_margin=0.7,
        #     sigmoid="linear",
        # )
        

        speed_comp = robot.center_of_mass_velocity()[0:2]
        if speed_comp[0] < 0 and speed_comp[1] < 0:
            speed_rew = 0
        else:
            speed = np.linalg.norm(speed_comp)
            speed_rew = rewards.tolerance(
                speed,
                bounds=(self._move_speed_lower_bound, self._move_speed_upper_bound),
                margin=self._move_speed_lower_bound,
                value_at_margin=0.1,
                sigmoid="gaussian",
            )
            if speed_comp[0] < 0 or speed_comp[1] < 0:
                speed_rew = min(speed_rew,0.5)

        move = move*speed_rew

        move = (5 * move + 1) / 6
        control_reward = (control_reward + 2) / 3
        reward = stand_reward*move*control_reward + avoid_obstacles


        #print("move",move,"control_reward",control_reward,"stand_reward",stand_reward)

        return reward, {
            "stand_reward": stand_reward,
            "small_control": control_reward,
            "move": move,
            "standing": standing,
            "upright": upright,
        }
    
def check_collision(env) -> bool:
    # check collision with the robot

    contacts = [env.data.contact[i] for i in range(env.data.ncon)]

    feet = ["left_ankle_link", "right_ankle_link"]
    feet_ids = [env.model.body(bn).id for bn in feet]
    for i,c in enumerate(contacts):
        geom1_body = env.model.body(env.model.geom_bodyid[c.geom1])
        geom2_body = env.model.body(env.model.geom_bodyid[c.geom2])
        # print("geom1_body",geom1_body)
        # print("geom2_body",geom2_body)
        geom1_is_floor = (env.model.body(geom1_body.rootid).name!="pelvis")
        geom2_is_foot = (env.model.geom_bodyid[c.geom2] in feet_ids)
        if not(geom1_is_floor and geom2_is_foot):
            return True
    return False