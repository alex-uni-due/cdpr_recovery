
from gym import Env
from gym.spaces import Space, Box
from typing import Optional, Union,  Callable, Literal, List, Dict #, Sequence
from abc import ABC, abstractmethod
import numpy as np
from numpy import array, cos, sin, cross
from numpy.linalg import norm
from cdpr.cdpr import *
from observations import *
from actions import *
from rewards import *
from tasks import *
from helper.utils import *
#%%   

class EmergencyStrategyCDPR(Env):
    r_p = property(lambda self: self.pose[:self.trans_dof])
    position = property(lambda self: self.r_p)
    r_p_dot = property(lambda self: self.pose[:self.trans_dof])
    velocity_vector = property(lambda self: self.r_p_dot)
    r_p_dot_dot = property(lambda self: self.pose_dot_dot[:self.trans_dof])
    acceleration_vector = property(lambda self: self.r_p_dot_dot)
    v = property(lambda self: norm(self.r_p_dot()))
    velocity = property(lambda self: self.v)
    a = property(lambda self: norm(self.r_p_dot_dot()))
    acceleration = property(lambda self: self.a)
     
    phi = property(lambda self: self.pose[-self.rot_dof:])
    phi_dot = property(lambda self: self.pose_dot[-self.rot_dof:])
    phi_dot_dot = property(lambda self: self.pose_dot_dot[-self.rot_dof:])
    omega = property(lambda self: norm(self.phi_dot()))
    omega_dot = property(lambda self: norm(self.phi_dot_dot()))
    orientation = property(lambda self: self.phi)
    angular_velocity = property(lambda self: self.omega)
    angular_velocity_vector = property(lambda self: self.phi_dot)
    angular_acceleration = property(lambda self: self.omega_dot)
    angular_acceleration_vector = property(lambda self: self.phi_dot_dot)
    
    cable_forces = property(lambda self: self.f)
    cable_force_gradients = property(lambda self: self.delta_f)
    # wrench = property(lambda self: self.w)
    wrench_external = property(lambda self: self.w_e)
    
    def __init__(self, 
                 cdpr: CDPR,
                 cables_idx: list, # list of cables to be used in simulation
                 Ts: float, # timestep intervall
                 coordinates: np.ndarray, # coordinate grid for workspace calculation
                 observation: Observation,
                 action: Action, # action function
                 reward: Reward, # reward function
                 task: Task,
                 *args,
                 **kwargs
                ) -> None:
                        
        # Parameters of the CDPR
        self.cdpr = cdpr
        self.cables_idx = cables_idx
        self.m = len(cables_idx)
        self.b = self.B[:,self.cables_idx] # remaining anchor points
        if self.P is None:
            self.p = None
        else:
            self.p = self.P[:,self.cables_idx] # remaining anchor points
        self.Ts = Ts # timestep
        self.coordinates = coordinates
        self.ws_full, self.ws_full_forces = calc_static_workspace(self.default_cables, coordinates)
        self.ws_rem, self.ws_rem_forces = calc_static_workspace(self.cables_idx, coordinates)
        
        if self.trans_dof==0:
            del self.r_p
            del self.r_p_dot
            del self.r_p_dot_dot
            del self.v
            del self.a
            del self.position
            del self.velocity
            del self.velocity_vector
            del self.acceleration
            del self.acceleration_vector
            
        if self.rot_dof==0:
            del self.phi
            del self.phi_dot
            del self.phi_dot_dot
            del self.omega 
            del self.omega_dot
            del self.orientation
            del self.angular_velocity
            del self.acceleration_vector
            del self.angular_acceleration
            del self.angular_acceleration_vector
            
        # Observation Space
        self._add_obs_limits(observation)
        self.get_obs = observation.func
        self.observation_space = observation.observation_space
        
        self.get_cable_foces = action.func
        self.action_space = action.action_space(self)
        self.get_reward = reward.func
        self.get_done = task.func
        self.success = task.success
        
    def _add_obs_limits(self,obs_vars):
        low = obs_vars.low
        high = obs_vars.high
        for i, var in enumerate (obs_vars.variables):
            if var.dtype==float or var.dtype==int:
                var_min = low[i]
                var_max = high[i]
                setattr(self, f"{var.symbol}_min", var_min)
                setattr(self, f"{var.symbol}_max", var_max)
            
    def __getattr__(self, attr):
        return getattr(self.cdpr, attr)
             
    ## Apply action to environment and observe new state and reward
    def step(self, action):
        self.steps += 1
        # Save previous states
        self.pose_prev = self.pose
        self.action_prev = self.act
        self.f_prev = self.f
        
        # Update states and actions
        self.f = self.get_cable_foces(action)
        self.delta_f = self.f - self.f_prev
        self.action = action
        self.pose_dot_dot = self.get_ode()
        self.pose, self.pose_dot = self.euler_cromer(self.pose_dot_dot)
        self.A = self.get_jacobian()
        
        # Get transition info
        self.state = self.get_obs()
        self.done = self.get_done()
        self.reward = self.get_reward()
        self.info = {"is_success": self.success}
        return self.state, self.reward, self.done, self.info
    
    def get_action(self):
        return rescale(self.f, self.f_min, self.f_max)
        
    def _default_obs(self):
        return np.hstack(self.pose.ravel(), 
                         self.pose_dot.ravel(), 
                         self.pose_dot_dot.ravel(),
                         self.f.ravel())
        
    def _default_action(self,action):
        return rescale(action, self.f_min, self.f_max)
    
    def reset(self,pose = None, pose_dot = None, pose_dot_dot=None):
        
        if pose is None:
           start_idx = np.random.randint(0, self.ws_full.shape[0])
           pose = (self.ws_full[start_idx]).reshape(self.n,1).astype(np.float32)
        if pose_dot is None:
            pose_dot = np.zeros((self.n,1))
        if pose_dot_dot is None:
            pose_dot_dot = np.zeros((self.n,1))
            
        self.steps = 0
        
        self.pose_0 = pose
        self.pose_prev = pose
        self.pose = pose
        
        self.pose_dot_0 = pose_dot
        self.pose_dot_prev = pose_dot
        self.pose_dot = pose_dot
        
        self.pose_dot_dot_0 = pose_dot_dot
        self.pose_dot_dot_prev = pose_dot_dot
        self.pose_dot_dot = pose_dot_dot
        
        A = self.cdpr.calc_jacobian(pose,self.m, self.n,self.B, self.P)
        self.A = A[self.cables,:] 
         
        self.f_0 = (quadprog(A,self.w, self.f_min, self.f_max)[self.cables]).reshape(self.m, 1)
        self.f_prev = self.f_0
        self.f = self.f_prev
        
        self.action_0 = normalize(self.f.ravel(), self.f_min, self.f_max)
        self.action_prev = self.action_0
        self.action = self.action_prev
        
        self.state = self.get_obs()
        self.done = False
        self.stopped = False
        self.collided = False
        
# %%
