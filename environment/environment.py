#%%
from gym import Env
import numpy as np
from numpy import array
from numpy.linalg import norm
from cdpr.cdpr import *
from environment.rewards import *
from environment.observations import *
from environment.actions import *
from helper.utils import *
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
    ## Properties
    #State
    spatial_velocity = property(lambda self: self.pose_dot)
    spatial_acceleration = property(lambda self: self.pose_dot_dot)
    cable_lengths = property(lambda self: self.a)
    structure_matrix = property(lambda self: self.get_structure_matrix())
    rotation_matrix = property(lambda self: self.get_rotmat())
    q = property(lambda self: self.get_cable_lenghts())
    cable_lengths = property(lambda self: self.q)
    
    r_p = property(lambda self: self.pose[:self.trans_dof])
    position = property(lambda self: self.r_p)
    r_p_dot = property(lambda self: self.pose[:self.trans_dof])
    velocity_vector = property(lambda self: self.r_p_dot)
    r_p_dot_dot = property(lambda self: self.pose_dot_dot[:self.trans_dof])
    acceleration_vector = property(lambda self: self.r_p_dot_dot)
    v = property(lambda self: norm(self.r_p_dot))
    velocity = property(lambda self: self.v)
    a = property(lambda self: norm(self.r_p_dot_dot))
    acceleration = property(lambda self: self.a)
     
    phi = property(lambda self: self.pose[-self.rot_dof:])
    phi_dot = property(lambda self: self.pose_dot[-self.rot_dof:])
    phi_dot_dot = property(lambda self: self.pose_dot_dot[-self.rot_dof:])
    omega = property(lambda self: norm(self.phi_dot))
    alpha = property(lambda self: norm(self.phi_dot_dot))
    orientation = property(lambda self: self.phi)
    angular_velocity = property(lambda self: self.omega)
    angular_velocity_vector = property(lambda self: self.phi_dot)
    angular_acceleration = property(lambda self: self.omega_dot)
    angular_acceleration_vector = property(lambda self: self.phi_dot_dot)
    
    cable_forces = property(lambda self: self.f)
    cable_force_gradients = property(lambda self: self.delta_f)
    # wrench = property(lambda self: self.w)
    wrench_external = property(lambda self: self.w_e)
    
    M_inv = property(lambda self: self.calc_mass_matrix_inverse())
    M = property(lambda self: self.calc_mass_matrix())
    
    def __init__(self, 
                 cdpr: CDPR,
                 cables_idx: list, # list of cables indeces to be used in simulation
                 Ts: float, # timestep intervall
                 coordinates: np.ndarray, # coordinate grid for workspace calculation and initialization
                 observation: Observation,
                 action: Action, # action function
                 reward: Reward, # reward function
                 task: Task,
                 **kwargs
                ) -> None:
                       
        # Parameters of the CDPR
        self.CDPR = cdpr
        self.cables_idx = cables_idx
        self.default_cables_idx = [i for i in range(len(self.default_cables))]
        self.m = len(cables_idx)
        self.b = self.B[:,self.cables_idx] # remaining anchor points
        if self.P is None:
            self.p = None
        else:
            self.p = self.P[:,self.cables_idx] # remaining anchor points
        self.Ts = Ts # timestep
        self.coordinates = coordinates
        self.ws_full, self.ws_full_forces = calc_static_workspace(self, self.default_cables_idx, coordinates)
        self.ws_rem, self.ws_rem_forces = calc_static_workspace(self, self.cables_idx, coordinates)
        not_ws = set(list(map(tuple, self.ws_full))).difference(set(list(map(tuple, self.ws_rem))))
        self.not_ws = array(list(not_ws))   
        for arg_name, arg_value in kwargs.items():
            setattr(self, arg_name, arg_value)
        
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
            del self.R
            del self.rotation_matrix
            
        # Observation Space
        self.observation = observation
        self._add_obs_limits()
        self.get_obs = lambda: observation.func(self)
        self.observation_space = observation.observation_space
        
        self.get_cable_foces = lambda act: action.func(self, act)
        self.action_space = action.action_space(self)
        self.get_reward = lambda: reward.func(self)
        self.is_success = lambda: task.is_success(self)
        self.is_failure = lambda: task.is_failure(self)
        self.is_timeout = lambda: task.is_timeout(self)
        
    def _add_obs_limits(self):
        low = self.observation.low
        high = self.observation.high
        i = 0
        for var in self.observation.variables:
            if var.dtype==float or var.dtype==int:
                var_min = low[i]
                var_max = high[i]
                setattr(self, f"{var.symbol}_min", var_min)
                setattr(self, f"{var.symbol}_max", var_max)
                i +=1
            elif var.dtype==np.ndarray:
                size = getattr(self, var.size)
                var_min = low[i:size]
                var_max = high[i:size]
                setattr(self, f"{var.symbol}_min", var_min)
                setattr(self, f"{var.symbol}_max", var_max)
                i+=size
                
            
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
        # Save previous states
        self.pose_prev = self.pose
        self.action_prev = self.action
        self.f_prev = self.f
        
        # Update states and actions
        self.f = self.get_cable_foces(action)
        self.delta_f = self.f - self.f_prev
        self.action = action
        self.pose_dot_dot = self.get_ode()
        self.pose_dot = euler_cromer(self.pose_dot, self.pose_dot_dot, self.Ts)
        self.pose = euler_cromer(self.pose, self.pose_dot, self.Ts)
        self.AT = self.get_structure_matrix()
        
        # Get transition info
        self.state = self.get_obs()
        self.done = self.get_done()
        self.reward = self.get_reward()
        self.info = {"is_success": self.success}
        return self.state, self.reward, self.done, self.info

    def get_done(self):
        self.success = self.is_success()
        self.failure = self.is_failure()
        self.timeout = self.is_timeout()
        return self.success or self.failure or self.timeout
    
    def get_action(self):
        return rescale(self.f, self.f_min, self.f_max)
        
    def get_state(self):
        state = []
        for i, var in enumerate(self.state):
            var_min = self.observation.low[i]
            var_max = self.observation.high[i]
            var = rescale(var, var_min, var_max)
            state.append(var)
        return array(state)
    
    def reset(self,pose = None, pose_dot = None, pose_dot_dot=None, f=None):
        
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
        self.done = False
        self.success = False
        self.failure = False
        self.timeout = False
        return self.state
# %%
