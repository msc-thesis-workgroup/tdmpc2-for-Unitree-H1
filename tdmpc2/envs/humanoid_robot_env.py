import os

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.envs import register
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from dm_control.mujoco import index
from dm_control.mujoco.engine import NamedIndexStructs
from scipy.spatial.transform import Rotation as R
# Local import
from .wrappers.dmc_wrapper import MjDataWrapper, MjModelWrapper
from .environment import Environment
from .utils import TrajectoryPlanner
from .utils import PositionController

from .robots import H1, H1Easy

from .rewards import (
    WalkV0,
    WalkV1,
    WalkV2,
    WalkV3,
    WalkV4,
    WalkV5,
    WalkV6,
    WalkV0Easy,
    WalkV2Easy,
    WalkV4Easy,
    WalkV5Easy
)
from .tasks import (
    Walk,
)


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 5.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": -20.0,
}
DEFAULT_RANDOMNESS = 0.01

DEFAULT_ENV_CONFIG = {
    "frame_skip": 10,
    "camera_name": "cam_default",
    "max_episode_steps": 1000
}

ROBOTS = {"h1": H1, 
          "h1Easy": H1Easy
          }

TASKS = {
    "walk": Walk
}

REWARDS = {
    "walk-v0": WalkV0,
    "walk-v1": WalkV1,
    "walk-v2": WalkV2,
    "walk-v3": WalkV3,
    "walk-v4": WalkV4,
    "walk-v5": WalkV5,
    "walk-v6": WalkV6,
    "walk-v0Easy": WalkV0Easy,
    "walk-v2Easy": WalkV2Easy,
    "walk-v4Easy": WalkV4Easy,
    "walk-v5Easy": WalkV5Easy
}

DEFAULT_TIME_STEP = 0.002
DEFAULT_COEFF = 0.25 # TODO: refactor this. It isn't good to have it in this module.

class HumanoidRobotEnv(MujocoEnv, gym.utils.EzPickle,Environment):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }
    # The parameters: robot, control, task are passed as kwargs to the __init__ function by make_env function in env_builder.py. All the possible environments are registered in env_builder.py before the make_env function is called.
    def __init__(
        self,
        robot=None,
        version=None,
        task=None,
        frame_skip = DEFAULT_ENV_CONFIG["frame_skip"],
        render_mode="rgb_array",
        width=256,
        height=256,
        randomness=DEFAULT_RANDOMNESS,
        **kwargs,
    ):
        #assert robot and control and task, f"{robot} {control} {task}"
        assert robot and task and version, f"{robot} {task} {version}"
        self.metadata["render_fps"] = 1 / (DEFAULT_TIME_STEP*frame_skip)
        gym.utils.EzPickle.__init__(self, metadata=self.metadata)

        # Go back to the previous directory
        asset_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        asset_path = os.path.join(asset_path, "asset")

        model_path = os.path.join(asset_path,robot.upper())
        model_path = os.path.join(model_path, f"scene.xml")

        print("[DEBUG: basic_locomotion_env] model_path:", model_path)
        print("[DEBUG: basic_locomotion_env] instance of robot:", robot, "instance of task:", task, "version:", version)

        self.robot = ROBOTS[robot]()
        self.task = TASKS[task]()
        self.task.set_observation_space(self.robot)

        self.obs_wrapper = kwargs.get("obs_wrapper", None)
        if self.obs_wrapper is not None:
            self.obs_wrapper = kwargs.get("obs_wrapper", "False").lower() == "true"
        else:
            self.obs_wrapper = False

        Environment.__init__(self)

        print("[DEBUG basic_locomotion_env]: frame_skip:", frame_skip)
        MujocoEnv.__init__(
            self,
            model_path,
            frame_skip= frame_skip,#DEFAULT_ENV_CONFIG["frame_skip"],
            observation_space=self.task.observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode=render_mode,
            width=width,
            height=height,
            camera_name=DEFAULT_ENV_CONFIG["camera_name"],
        )
        print("[DEBUG basic_env_elements]: timestep:",self.model.opt.timestep)
        print("[DEBUG basic_env_elements]: dt:",self.dt)
        # Setting up the action space
        self.action_high = self.action_space.high
        self.action_low = self.action_space.low
        self.action_space = Box(
            low=-1, high=1, shape=self.action_space.shape, dtype=np.float32
        )

        self.observation_space = self.task.observation_space

        print("[DEBUG basic_locomotion_env] self.observation_space:", self.observation_space)
        print("[DEBUG basic_locomotion_env] self.action_space:", self.action_space)

        # Keyframe
        self.keyframe = (
            self.model.key(kwargs["keyframe"]).id if "keyframe" in kwargs else 0
        )

        self.randomness = randomness

        # Set up named indexing.
        data = MjDataWrapper(self.data)
        model = MjModelWrapper(self.model)
        axis_indexers = index.make_axis_indexers(model)
        self.named = NamedIndexStructs(
            model=index.struct_indexer(model, "mjmodel", axis_indexers),
            data=index.struct_indexer(data, "mjdata", axis_indexers),
        )

        self.robot.update_robot_state(self)
        self.task.set_reward(REWARDS[f"{task}-{version}"](robot=self.robot))

        assert self.robot.dof + self.task.dof == len(self.data.qpos), (
            self.robot.dof,
            self.task.dof,
            len(self.data.qpos),
        )

        self.controller = PositionController(self.robot, coeff=DEFAULT_COEFF) # TODO: refactor DEFAULT_COEFF. It isn't good to have it in this module.

    def get_joint_torques(self,ctrl):

        # TODO(my-rice): I need to change this. The kp and kd values are hard coded. I need to take them from the specific robot class. (from robots.py)
        # kp = np.array([200, 200, 200, 300, 40, 200, 200, 200, 300, 40, 300, 100, 100, 100, 100, 100, 100, 100, 100])
        # kd = np.array([5, 5, 5, 6, 2, 5, 5, 5, 6, 2, 6, 2, 2, 2, 2, 2, 2, 2, 2])

        kp = self.robot.get_kp()
        kd = self.robot.get_kd()

        actuator_length = self.data.qpos[7:26] # self.data.actuator_length
        error = ctrl - actuator_length
        m = self.model
        d = self.data

        empty_array = np.zeros(m.actuator_dyntype.shape)

        ctrl_dot = np.zeros(m.actuator_dyntype.shape) if np.array_equal(m.actuator_dyntype,empty_array) else d.act_dot[m.actuator_actadr + m.actuator_actnum - 1]

        error_dot = ctrl_dot - self.data.qvel[6:26] # self.data.actuator_velocity

        joint_torques = kp*error + kd*error_dot

        return joint_torques




    def step(self, action):

        #TODO refactor this. I don't like the fact that these steps are made here. I would like to have them in the task class. Environment should only be responsible for the simulation.
        action_high = self.robot.get_upper_limits()
        action_low = self.robot.get_lower_limits()

        desired_joint_position = (action + 1) / 2 * (action_high - action_low) + action_low


        #TODO implement a switch to choose between the two techniques in configuration file

        # SOLUZIONE 1
        #action = self.get_joint_torques(desired_joint_position)
        # action = self.controller.control_step(self.model, self.data, desired_joint_position, np.zeros_like(self.data.qvel[6:]))
        # self.do_simulation(action, self.frame_skip)

        # SOLUZIONE 2
        for _ in range(self.frame_skip):
            torque_reference = self.controller.control_step2(self.model, self.data, desired_joint_position)
            #torque_reference = self.controller.control_step(self.model, self.data, desired_joint_position, np.zeros_like(self.data.qvel[6:]))
            self.do_simulation(torque_reference, 1)

        # SOLUZIONE 3
        # t = 0
        # duration = self.frame_skip*self.model.opt.timestep
        # vel_reference = np.zeros_like(self.data.qvel[6:])
        # traj_planner = TrajectoryPlanner(starting_qpos=self.data.qpos[7:], starting_qvel=vel_reference, duration=duration, final_qpos=desired_joint_position, final_qvel=vel_reference)
        # for _ in range(self.frame_skip):
        #     t = t + self.model.opt.timestep
        #     desired_pos = traj_planner.get_pos(t)
        #     desired_vel = traj_planner.get_vel(t)
        #     torque_reference = self.controller.control_step(self.model, self.data, desired_pos, desired_vel)
        #     self.do_simulation(torque_reference, 1)

        #print("[DEBUG basic_locomotion_env]: qpos error:", self.data.qpos[7:26] - desired_joint_position)

        #print("[DEBUG basic_locomotion_env]: env.data.time:", self.data.time)
        obs = self.get_obs()
        self.robot.update_robot_state(self)
        #reward, reward_info = self.task.get_reward(self.robot, torque_reference)
        reward, reward_info = self.task.get_reward(self.robot, desired_joint_position)

        #print("[DEBUG basic_locomotion_env]: error:", np.mean(np.abs(self.data.qpos[7:26] - desired_joint_position)))
        #print("[DEBUG basic_locomotion_env]: velocity:", self.data.qvel)
        #print("[DEBUG basic_locomotion_env]: self.data.qpos[2]:", self.data.qpos[2])

        terminated, terminated_info = self.task.get_terminated(self)

        info = {"per_timestep_reward": reward, **reward_info, **terminated_info}
        return obs, reward, terminated, False, info

    def get_obs(self):
        return self.task.get_obs(env=self)

    def mock_next_state(self, action):
        # TODO delete this function
        return self.task.mock_next_state(action)

    def reset_model(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.keyframe)
        mujoco.mj_forward(self.model, self.data)

        # Add randomness
        init_qpos = self.data.qpos.copy()
        init_qvel = self.data.qvel.copy()
        r = self.randomness
        self.set_state(
            init_qpos + self.np_random.uniform(-r, r, size=self.model.nq), init_qvel
        )

        # Task-specific reset and return observations
        return self.task.reset_model(self)

    def seed(self, seed=None):
        np.random.seed(seed)

    def render(self):
        return self.mujoco_renderer.render(
            self.render_mode, self.camera_id, self.camera_name
        )
