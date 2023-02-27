
#%%
import numpy as np
from typing import Optional, Union, Callable, Sequence
from gym.spaces import Box
from cdpr.utils import normalize
from environment import states

class ObservationFunction:
    def __init__(
        self,
        func: Callable,
        variables: Sequence[Union[states.StateVariable,states.DerivedVariable]]
        ) -> None:
        """Funtion for observing state of an environment

        Parameters
        ----------
        func : Callable
            function that calls observations
        variables : Sequence[Union[states.StateVariable,states.DerivedVariable]]
            variables that are observed
        """
        self.variables = variables
        self.func = func
        self.__doc__ = func.__doc__
    
    def __call__(self,*args, **kwargs):
        return self.func(*args, **kwargs)
    def __str__(self) -> str:
        return str(self.__dict__)
    def __repr__(self) -> str:
        return str(self.__dict__) 
            
class Observation:
    def __init__(
        self,
        obsfunc: ObservationFunction,
        low: Union[np.ndarray, Sequence[float]] ,
        high: Union[np.ndarray, Sequence[float]],
        normalized: bool = True,
        description: str = ""):
        """Class for defining varibales that should be observed during CDPR Simulation

        Parameters
        ----------
        variables : List[Union[StateVariable,DerivedVariable]]
            List of varibles to observe. Must be present in states module.
        func : Callable
            Function that delivers the state.
        low : Union[np.ndarray, List[float]]
            Minimum values for observation variables
        high : Union[np.ndarray, List[float]]
            Maximum values for observartion variables
        normalized : bool, optional
            Normalize the observarion variables to the interval [-1,1], by default True
        description : str, optional
            Text providing additional information, by default ""
        """
        self.variables = obsfunc.variables
        assert len(low)==len(high), "length of low and high must be equal"
        self.low = np.array(low) if type(low)!=np.ndarray else low
        self.high = np.array(high) if type(high)!=np.ndarray else high
        self.normalized = normalized
        self._func = obsfunc.func
        if normalized:
            self.func = self.normalized_func
            self.observation_space = Box(-1,1,(len(low),))
        else:
            self.func = self.unnormalized_func
            self.observation_space = Box(low, high)
        if description == "":
            description = str([var.name for var in obsfunc.variables])
        self.description = description
        
    def unnormalized_func(self, cdpr):
        return self._func(cdpr).astype("float32")
    def normalized_func(self, cdpr):
        return normalize(self._func(cdpr), self.low, self.high).astype("float32")
    def __str__(self) -> str:
        return str(self.__dict__)
    def __repr__(self) -> str:
        return str(self.__dict__)
   
def _obsfunc_0(env):
    """Observe [pose, spatial_velocities, cable_forces]"""
    return np.vstack((env.pose, 
                     env.pose_dot, 
                     env.f)).ravel()
_obsvars_0 = [states.pose, states.spatial_velocity, states.cable_forces]
obsfunc0 = ObservationFunction(_obsfunc_0, _obsvars_0)

def _obsfunc_1(env):
    """Observe [pose, velocities, cable_forces, position_error]"""
    env.position_error = env.goal-env.r_p
    env.distance = np.linalg.norm(env.position_error)
    return np.vstack((env.pose, 
                     env.pose_dot, 
                     env.f,
                     env.position_error)).ravel()  
_obsvars_1 = [states.pose, states.spatial_velocity, states.cable_forces, states.position_error]
obsfunc1 = ObservationFunction(_obsfunc_1, _obsvars_1)

def _obsfunc_2(env):
    """Observe [pose, spatial_velocities, spatial_accelerations, cable_forces]"""
    return np.vstack((env.pose, 
                     env.pose_dot, 
                     env.pose_dot_dot,
                     env.f)).ravel()
_obsvars_2 = [states.pose, states.spatial_velocity, states.acceleration, states.cable_forces]
obsfunc2 = ObservationFunction(_obsfunc_2, _obsvars_2)
    
def _obsfunc_3(env):
    """Observe [pose, velocities, cable_forces, cable_lengths]"""
    return np.vstack((env.pose, 
                     env.pose_dot, 
                     env.f,
                     env.q)).ravel()
_obsvars_3 = [states.pose, states.spatial_velocity, states.cable_forces, states.cable_lengths]
obsfunc3 = ObservationFunction(_obsfunc_3, _obsvars_3)
 
def _obsfunc_4(env):
    """Observe [pose, velocities, cable_forces, steps]"""
    return np.vstack((env.pose, 
                     env.pose_dot, 
                     env.f,
                     env.steps)).ravel()
_obsvars_4 = [states.pose, states.spatial_velocity, states.cable_forces, states.steps]
obsfunc4 = ObservationFunction(_obsfunc_2, _obsvars_4)
      
# %%
