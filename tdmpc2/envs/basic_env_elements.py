# This file is a modified version of the original file from the HumanoidBench repository
import os

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.envs import register
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from dm_control.mujoco import index
from dm_control.mujoco.engine import NamedIndexStructs
from dm_control.utils import rewards

# Local import
from .wrappers.dmc_wrapper import MjDataWrapper, MjModelWrapper

from .wrappers.wrappers import (
    ObservationWrapper
)

from .robots import H1

from .tasks.basic_locomotion_tasks import (
    Stand,
    Walk,
    Run
)


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 5.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": -20.0,
}
DEFAULT_RANDOMNESS = 0.01

ROBOTS = {"h1": H1} 

TASKS = {
    "stand": Stand,
    "walk": Walk,
    "run": Run
}

class HumanoidEnv(MujocoEnv, gym.utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    # As far as I understand, the parameters robot, control, task are passed as kwargs to the __init__ function by the register function.
    def __init__(
        self,
        robot=None,
        control=None,
        task=None,
        render_mode="rgb_array",
        width=256,
        height=256,
        randomness=DEFAULT_RANDOMNESS,
        **kwargs,
    ):
        assert robot and control and task, f"{robot} {control} {task}"
        gym.utils.EzPickle.__init__(self, metadata=self.metadata)
        
        # Go back to the previous directory
        asset_path = os.path.dirname(os.path.dirname(__file__))

        asset_path = os.path.join(asset_path, "assets")

        model_path = f"envs/{robot}_{control}_{task}.xml"
        model_path = os.path.join(asset_path, model_path)
        model_path = "/home/davide/tdmpc2/tdmpc2/unitree_h1/scene.xml" #TODO(my-rice) fix this. You need to change dynamically the path. It must not be hard coded.
        print("[DEBUG: basic_locomotion_env] model_path:", model_path)
        self.robot = ROBOTS[robot](self)
        task_info = TASKS[task](self.robot, None, **kwargs)

        self.obs_wrapper = kwargs.get("obs_wrapper", None)
        if self.obs_wrapper is not None:
            self.obs_wrapper = kwargs.get("obs_wrapper", "False").lower() == "true"
        else:
            self.obs_wrapper = False

        MujocoEnv.__init__(
            self,
            model_path,
            frame_skip=task_info.frame_skip,
            observation_space=task_info.observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            render_mode=render_mode,
            width=width,
            height=height,
            camera_name=task_info.camera_name,
        )
        #print("[DEBUG basic_env_elements]: timestep:",self.model.opt.timestep)
        self.action_high = self.action_space.high
        self.action_low = self.action_space.low
        self.action_space = Box(
            low=-1, high=1, shape=self.action_space.shape, dtype=np.float32
        )

        self.task = TASKS[task](self.robot, self, **kwargs)

        # Wrap for hierarchical control ### NOTE: I think they are not needed for the basic locomotion tasks.
        if (
            "policy_type" in kwargs
            and kwargs["policy_type"]
            and kwargs["policy_type"] is not None
            and kwargs["policy_type"] != "flat"
        ):
            raise NotImplementedError("Hierarchical policy is not supported yet.")
        

            # if kwargs["policy_type"] == "reach_single":
            #     assert "policy_path" in kwargs and kwargs["policy_path"] is not None
            #     self.task = SingleReachWrapper(self.task, **kwargs)
            # elif kwargs["policy_type"] == "reach_double_absolute":
            #     assert "policy_path" in kwargs and kwargs["policy_path"] is not None
            #     self.task = DoubleReachAbsoluteWrapper(self.task, **kwargs)
            # elif kwargs["policy_type"] == "reach_double_relative":
            #     assert "policy_path" in kwargs and kwargs["policy_path"] is not None
            #     self.task = DoubleReachRelativeWrapper(self.task, **kwargs)
            # elif kwargs["policy_type"] == "blocked_hands":
            #     self.task = BlockedHandsLocoWrapper(self.task, **kwargs)
            # else:
            #     raise ValueError(f"Unknown policy_type: {kwargs['policy_type']}")
        elif self.obs_wrapper: # NOTE: I don't think this is needed for the basic locomotion tasks. But I will keep it for now. #TODO(my-rice) remove this if it is not needed.
            # Note that observation wrapper is not compatible with hierarchical policy
            self.task = ObservationWrapper(self.task, **kwargs)
            self.observation_space = self.task.observation_space

        # Keyframe
        self.keyframe = (
            self.model.key(kwargs["keyframe"]).id if "keyframe" in kwargs else 0
        )

        self.randomness = randomness

        ### NOTE: I will not work with Kitchen, Bookshelf and so on. I will work with the basic locomotion tasks.
        # if isinstance(self.task, (BookshelfHard, BookshelfSimple, Kitchen, Cube)):
        #     self.randomness = 0
        # print(isinstance(self.task, (BookshelfHard, BookshelfSimple, Kitchen, Cube)))

        # Set up named indexing.
        data = MjDataWrapper(self.data)
        model = MjModelWrapper(self.model)
        axis_indexers = index.make_axis_indexers(model)
        self.named = NamedIndexStructs(
            model=index.struct_indexer(model, "mjmodel", axis_indexers),
            data=index.struct_indexer(data, "mjdata", axis_indexers),
        )

        assert self.robot.dof + self.task.dof == len(data.qpos), (
            self.robot.dof,
            self.task.dof,
            len(data.qpos),
        )

    def get_joint_torques(self,ctrl):
        
        # TODO(my-rice): I need to change this. The kp and kd values are hard coded. I need to take them from the specific robot class. (from robots.py)
        kp = np.array([200, 200, 200, 300, 40, 200, 200, 200, 300, 40, 300, 100, 100, 100, 100, 100, 100, 100, 100])
        kd = np.array([5, 5, 5, 6, 2, 5, 5, 5, 6, 2, 6, 2, 2, 2, 2, 2, 2, 2, 2])
        

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

        return self.task.step(action)


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
        return self.task.reset_model()

    def seed(self, seed=None):
        np.random.seed(seed)

    def render(self):
        return self.task.render()
