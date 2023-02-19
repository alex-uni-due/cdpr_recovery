
#%%
import numpy as np
from typing import Optional, Union, Callable, Sequence
from gym.spaces import Box
from helper.utils import normalize
from environment.states import *
        
class Observation:
    def __init__(
        self,
        variables: Sequence[Union[StateVariable,DerivedVariable]],
        func: Callable,
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
        self.variables = variables
        assert len(low)==len(high), "length of low and high must be equal"
        self.low = np.array(low) if type(low)!=np.ndarray else low
        self.high = np.array(high) if type(high)!=np.ndarray else high
        self.normalized = normalized
        if normalized:
            self.func = lambda cdpr: normalize(func(cdpr), self.low, self.high).astype("float32")
            self.observation_space = Box(-1,1,(len(low),))
        else:
            self.func = lambda cdpr: func(cdpr).astype("float32")
            self.observation_space = Box(low, high)
        if description == "":
            description = str([var.name for var in variables])
        self.description = description
    def __str__(self) -> str:
        return str(self.__dict__)
    def __repr__(self) -> str:
        return str(self.__dict__)
    
def obs_func_0(env):
    """Observe [pose, spatial_velocities, forces]"""
    return np.vstack(env.pose, 
                     env.pose_dot, 
                     env.f).ravel()
    
def obs_func_1(env):
    """Observe [pose, spatial_velocities, spatial_accelerations, forces]"""
    return np.vstack(env.pose, 
                     env.pose_dot, 
                     env.pose_dot_dot,
                     env.f).ravel()

def obs_func_2(env):
    """Observe [pose, velocities, forces, steps]"""
    return np.vstack(env.pose, 
                     env.pose_dot, 
                     env.f,
                     env.steps).ravel()
    
def obs_func_3(env):
    """Observe [pose, velocities, forces, q]"""
    return np.vstack(env.pose, 
                     env.pose_dot, 
                     env.f,
                     env.q).ravel()
       
def obs_func_4(env):
    """Observe [pose, velocities, forces, I]"""
    return np.vstack(env.pose, 
                     env.pose_dot, 
                     env.f,
                     env.I).ravel()  
# %%
