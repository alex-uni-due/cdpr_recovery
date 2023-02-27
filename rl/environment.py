#%%
from gym import Env
import numpy as np
from numpy import array
from numpy.linalg import norm
from cdpr.cdpr import *
from .rewards import *
from .observations import *
from .actions import *
from cdpr.utils import *
#%%
class Task:
    def __init__(self,
                 success_func, 
                 failure_func,
                 timeout_func,
                 description) -> None:
        self.is_success = success_func
        self.is_failure = failure_func
        self.is_timeout = timeout_func
        self.description = description
    def __str__(self) -> str:
        return str(self.__dict__)
    def __repr__(self) -> str:
        return str(self.__dict__)
#%%
class EmergencyStrategyCDPR(Env):
    #%% Properties
    @property
    def spatial_velocity(self): return self.pose_dot
    @property
    def spatial_acceleration(self): return self.pose_dot_dot
    @property
    def cable_lengths(self): return self.a
    @property
    def structure_matrix(self): return self.get_structure_matrix()
    @property
    def rotation_matrix(self): return self.get_rotmat()
    @property
    def q(self): return self.get_cable_lenghts()
    @property
    def cable_lengths(self): return self.q
    @property
    def r_p(self): return self.pose[:self.trans_dof]
    @property
    def position(self): return self.r_p
    @property
    def r_p_dot(self): return self.pose[:self.trans_dof]
    @property
    def velocity_vector(self): return self.r_p_dot
    @property
    def r_p_dot_dot(self): return self.pose_dot_dot[:self.trans_dof]
    @property
    def acceleration_vector(self): return self.r_p_dot_dot
    @property
    def v(self): return norm(self.r_p_dot)
    @property
    def velocity(self): return self.v
    @property
    def a(self): return norm(self.r_p_dot_dot)
    @property
    def acceleration(self): return self.a
    @property
    def phi(self): return self.pose[-self.rot_dof:]
    @property
    def phi_dot(self): return self.pose_dot[-self.rot_dof:]
    @property
    def phi_dot_dot(self): return self.pose_dot_dot[-self.rot_dof:]
    @property
    def omega(self): return norm(self.phi_dot)
    @property
    def alpha(self): return norm(self.phi_dot_dot)
    @property
    def orientation(self): return self.phi
    @property
    def angular_velocity(self): return self.omega
    @property
    def angular_velocity_vector(self): return self.phi_dot
    @property
    def angular_acceleration(self): return self.omega_dot
    @property
    def angular_acceleration_vector(self): return self.phi_dot_dot
    @property
    def cable_forces(self): return self.f
    @property
    def cable_force_gradients(self): return self.del_ta_f
    @property
    def wrench_external(self): return self.w_e
    @property
    def M_inv(self): return self.calc_mass_matrix_inverse()
    @property
    def M(self): return self.calc_mass_matrix()
    
    #%%
    def __init__(self, 
                 cdpr: CDPR,
                 cables_idx: List[int], # list of cables indeces to be used in simulation
                 Ts: float, # timestep intervall
                 max_episode_timesteps: int,
                 coordinates: np.ndarray, # coordinate grid for workspace calculation and initialization
                 observation: Observation,
                 action: Action, # action function
                 reward: Reward, # reward function
                 task: Task,
                 rew_params:Dict[str,float] = {},
                 **kwargs
                ) -> None:
                       
        # Parameters of the CDPR
        self.CDPR = cdpr
        self.cables_idx = cables_idx
        self.default_cables_idx = list(cdpr.cables.index_mapping.keys())
        self.m = len(cables_idx)
        self.b = self.B[:,self.cables_idx] # remaining anchor points
        if self.P is None:
            self.p = None
        else:
            self.p = self.P[:,self.cables_idx] # remaining anchor points
        self.Ts = Ts # timestep
        self.max_episode_timesteps = max_episode_timesteps
        self.coordinates = coordinates
        self.ws_full, self.ws_full_forces = calc_static_workspace(self, coordinates, self.default_cables_idx)
        self.ws_rem, self.ws_rem_forces = calc_static_workspace(self, coordinates, self.cables_idx)
        not_ws = set(list(map(tuple, self.ws_full))).difference(set(list(map(tuple, self.ws_rem))))
        self.not_ws = array(list(not_ws))   
        for arg_name, arg_value in kwargs.items():
            setattr(self, arg_name, arg_value)
        
        # Observation Space
        self.obs = observation
        self._add_obs_limits()
        # self.get_obs = lambda: observation.func(self)
        self.observation_space = observation.observation_space
        
        self.act = action
        # self.get_cable_forces = lambda act: action.func(self, act)
        if callable(action.action_space):
            self.action_space = action.action_space(self)
        else:
            self.action_space = action.action_space
        self.rew = reward
        self.rew_params = rew_params
        # self.get_reward = lambda: reward.func(self)
        self.task = task
        # self.is_success = lambda: task.is_success(self)
        # self.is_failure = lambda: task.is_failure(self)
        # self.is_timeout = lambda: task.is_timeout(self)
    
    def get_obs(self):
        return self.obs.func(self)
    
    def get_cable_forces(self, action):
        return self.act.func(self,action)
    
    def get_reward(self):
        return self.rew.func(self, **self.rew_params)
            
    def set_rew(self, reward:Reward):
        self.rew = reward
    def set_rew_params(self, params):
        self.rew_params = params
    def set_act(self, action:Action):
        self.act = action
        self.action_space = action.action_space(self)
        
    def set_obs(self, observation:Observation):
        self.obs = observation
        self._add_obs_limits()
        self.observation_space = observation.observation_space
        
    def set_task(self, task:Task):
        self.task = task
            
    def is_success(self):
        return self.task.is_success(self)
    
    def is_failure(self):
        return self.task.is_failure(self)
    
    def is_timeout(self):
        return self.task.is_timeout(self,self.max_episode_timesteps)
    
    def _add_obs_limits(self):
        """Adds limits variable names and their limits as attributes"""
        low = self.obs.low
        high = self.obs.high
        i = 0
        for var in self.obs.variables:
            if var.dtype==float or var.dtype==int:
                if not hasattr(self,f"{var.symbol}_min"):
                    var_min = low[i]
                    setattr(self, f"{var.symbol}_min", var_min)
                if not hasattr(self,f"{var.symbol}_max"):
                    var_max = high[i]
                    setattr(self, f"{var.symbol}_max", var_max)
                i +=1
            elif var.dtype==np.ndarray:
                size = getattr(self, var.size)
                if not hasattr(self,f"{var.symbol}_min"):
                    var_min = low[i:size]
                    setattr(self, f"{var.symbol}_min", var_min)
                if not hasattr(self,f"{var.symbol}_max"):    
                    var_max = high[i:size]
                    setattr(self, f"{var.symbol}_max", var_max)
                i+=size
                
    # Function for accessing cdpr attrs        
    def __getattr__(self, attr):
        return getattr(self.CDPR, attr)
    
    def get_structure_matrix(self):
        return self.CDPR.calc_structure_matrix(self.pose, self.b, self.p)
    
    def get_rotmat(self):
        return self.CDPR.calc_rotmat(self.pose)
    
    def get_cable_lengths(self):
        return self.CDPR.calc_ode(self.pose, self.m, self.b, self.p)
    
    def get_ode(self):
        return self.CDPR.calc_ode(self.M_inv, self.AT, self.f, self.w_e)
    
    ## Apply action to environment and observe new state and reward
    def step(self, action):
        self.steps += 1
        # Save previous states and actions for potential use with gradient functions
        self.state_prev = self.state
        self.action_prev = self.action
        self.pose_prev = self.pose
        self.pose_dot_prev = self.pose_dot
        self.pose_dot_dot_prev = self.pose_dot_dot
        self.f_prev = self.f
        
        # Update states and actions
        self.f = self.get_cable_forces(action)
        self.delta_f = self.f - self.f_prev
        self.action = action
        self.pose_dot_dot = self.CDPR.calc_ode(self.M_inv, self.AT, self.f, self.w_e)
        self.pose_dot = euler_cromer(self.pose_dot, self.pose_dot_dot, self.Ts)
        self.pose = euler_cromer(self.pose, self.pose_dot, self.Ts)
        self.AT = self.calc_structure_matrix(self.pose, self.b, self.p)
        
        # Get transition info
        self.state = self.get_obs()
        self.success = self.is_success()
        self.failure = self.is_failure()
        self.timeout = self.is_timeout()
        self.done = self.get_done()
        self.reward = self.get_reward()
        self.info = {"is_success": self.success}
        return self.state, self.reward, self.done, self.info

    def get_done(self):
        return self.success or self.failure or self.timeout
    
    def get_rescaled_state(self):
        state = []
        for i, var in enumerate(self.state):
            var_min = self.obs.low[i]
            var_max = self.obs.high[i]
            var = rescale(var, var_min, var_max)
            state.append(var)
        state = array(state)
        return state
    
    def seed(self, seed):
        np.random.seed(seed)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        
    def reset(self,pose = None, 
              pose_dot = None, 
              pose_dot_dot=None, 
              f=None):
        
        if pose is None:
           start_idx = np.random.randint(0, self.ws_full.shape[0])
           pose = (self.ws_full[start_idx]).reshape(self.n,1).astype(np.float32)
           
        if pose_dot is None:
            pose_dot = np.zeros((self.n,1))
        if pose_dot_dot is None:
            pose_dot_dot = np.zeros((self.n,1))
            pose_dot_dot[1] = -self.m_P*9.81 
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
        
        AT = self.calc_structure_matrix(pose,self.B, self.P)
        self.AT = AT[:,self.cables_idx] 
        if f is None:
            self.f_0 = (quadprog(AT,self.w_e, self.f_min, self.f_max)[self.cables_idx]).reshape(self.m, 1)
        else:
            self.f_0 = f
            
        self.f_prev = self.f_0
        self.f = self.f_prev
        
        self.action_0 = normalize(self.f.ravel(), self.f_min, self.f_max)
        self.action_prev = self.action_0
        self.action = self.action_prev
        
        self.state = self.get_obs()
        self.state_prev = self.state
        self.done = False
        self.success = False
        self.failure = False
        self.timeout = False
        return self.state
# %%
