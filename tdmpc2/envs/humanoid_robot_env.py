import os

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.envs import register
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from dm_control.mujoco import index
from dm_control.mujoco.engine import NamedIndexStructs

# Local import
from .wrappers.dmc_wrapper import MjDataWrapper, MjModelWrapper
from .environment import Environment

from .robots import H1

from .rewards import (
    WalkV0,
    WalkV1,
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

ROBOTS = {"h1": H1} 

TASKS = {
    "walk": Walk
}

REWARDS = {
    "walk-v0": WalkV0,
    "walk-v1": WalkV1,
}

DEFAULT_TIME_STEP = 0.002

class HumanoidRobotEnv(MujocoEnv, gym.utils.EzPickle,Environment):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }
    # As far as I understand, the parameters robot, control, task are passed as kwargs to the __init__ function by the register function.
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

    def get_joint_torques(self,ctrl):
        
        # TODO(my-rice): I need to change this. The kp and kd values are hard coded. I need to take them from the specific robot class. (from robots.py)
        # kp = np.array([200, 200, 200, 300, 40, 200, 200, 200, 300, 40, 300, 100, 100, 100, 100, 100, 100, 100, 100])
        # kd = np.array([5, 5, 5, 6, 2, 5, 5, 5, 6, 2, 6, 2, 2, 2, 2, 2, 2, 2, 2])
        
        kp = self.robot.get_kp()
        kd = self.robot.get_kd()

        self.data.ctrl = ctrl

        actuator_length = self.data.actuator_length
        error = ctrl - actuator_length
        m = self.model
        d = self.data
        
        empty_array = np.zeros(m.actuator_dyntype.shape)
        
        ctrl_dot = np.zeros(m.actuator_dyntype.shape) if np.array_equal(m.actuator_dyntype,empty_array) else d.act_dot[m.actuator_actadr + m.actuator_actnum - 1]
        
        error_dot = ctrl_dot - self.data.actuator_velocity
        
        joint_torques = kp*error + kd*error_dot

        return joint_torques

    def step(self, action):
        
        #TODO refactor this. I don't like the fact that these steps are made here. I would like to have them in the task class. Environment should only be responsible for the simulation.
        action_high = self.robot.get_upper_limits()
        action_low = self.robot.get_lower_limits()

        desired_joint_position = (action + 1) / 2 * (action_high - action_low) + action_low
        
        action = self.get_joint_torques(desired_joint_position)
        
        self.do_simulation(action, self.frame_skip)

        #print("[DEBUG basic_locomotion_env]: env.data.time:", self.data.time)
        obs = self.get_obs()
        self.robot.update_robot_state(self)
        reward, reward_info = self.task.get_reward(self.robot, action)
        terminated, terminated_info = self.task.get_terminated(self)

        info = {"per_timestep_reward": reward, **reward_info, **terminated_info}
        return obs, reward, terminated, False, info

    def get_obs(self):
        return self.task.get_obs(env=self)

    def mock_next_state(self, action):
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
