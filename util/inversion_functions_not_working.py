# This file contains the functions that are used to invert the dynamics of the MuJoCo model.
# However, the functions are not working as expected and are not used in the current implementation.
# This file is kept for reference and for future work. (If needed)

#     def invert_dynamics_2(self, action,action_torque):
#         data_copy = copy.deepcopy(self.data)
#         data_copy.ctrl[:] = action_torque

#         mujoco.mj_step(self.model, data_copy, nstep=self.frame_skip)
#         mujoco.mj_rnePostConstraint(self.model, data_copy)
#         #print("qacc[0:6]:", data_copy.qacc[0:6],"test:", (data_copy.qvel[0:6] - self.data.qvel[0:6])/(data_copy.time-self.data.time))
#         old_acc = self.data.qacc.copy()

#         self.data.qacc = data_copy.qacc.copy()

#         mujoco.mj_inverse(self.model, self.data)

#         self.data.qacc = old_acc
#         sol = self.data.qfrc_inverse.copy()
#         self.data.ctrl = sol[6:25]

#         mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
#         mujoco.mj_rnePostConstraint(self.model, self.data)


#         #print("[DEBUG basic_locomotion_env]: VELOCITY ERROR:",self.data.qvel - data_copy.qvel)
#         #print("[DEBUG basic_locomotion_env]: POSITION ERROR:",self.data.qpos - data_copy.qpos)
#         #print("[DEBUG basic_locomotion_env]: data.solver_fwdinv:",self.data.solver_fwdinv)
#         del data_copy

#         return action_torque
#     def do_simulation_by_inversion(self, action, n_frames):

#         """
#         Step the simulation n number of frames and applying a control action by inverting the dynamics.
#         """
#         # Check control input is contained in the action space
#         if np.array(action).shape != (self.model.nu,):
#             raise ValueError(
#                 f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(action).shape}"
#             )
        
#         initial_qpos = self.data.qpos.copy()[7:26]
#         initial_qvel = self.data.qvel.copy()[6:26]

#         final_qpos = action
#         final_qvel = np.zeros_like(initial_qvel) 

#         duration = n_frames * self.model.opt.timestep # = self.dt
#         traj_planner = TrajectoryPlanner(starting_qpos=initial_qpos, starting_qvel=initial_qvel, duration=duration, final_qpos=final_qpos, final_qvel=final_qvel)
#         t = 0.0
#         for _ in range(n_frames):

            
#             desired_acc = traj_planner.get_acc(t)

#             # inverting
#             old_acc = self.data.qacc.copy()

#             self.data.qacc[6:25] = desired_acc
#             self.data.qacc[0:6] = np.zeros(6)

#             mujoco.mj_inverse(self.model, self.data)

#             self.data.qacc = old_acc

#             sol = self.data.qfrc_inverse.copy()
#             self.data.ctrl = sol[6:25]

#             mujoco.mj_step(self.model, self.data, nstep=1)

#             t += self.model.opt.timestep
            
