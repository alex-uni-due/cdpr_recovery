
#%%
import numpy as np
from typing import Optional, Union, List, Callable#, Sequence
from gym.spaces import Box, Space
from helper.utils import normalize, rescale
from cdpr.states import *
#%%
# @dataclass(frozen=True, order=True)
class Observation:
    """Dataclass for defining varibales that should be observed during CDPR Simulation
    """
    # observations: List[str]
    # size: int
    # observation_space: Space
    # normalized: bool = True
    # description: Optional[str] = field(default_factory=str)
    
    def __init__(
        self,
        variables: List[Union[StateVariable,DerivedVariable]],
        func: Callable,
        low: Union[np.ndarray, List[float]] ,
        high: Union[np.ndarray, List[float]],
        normalized: bool = True,
        description: Optional[str] = ""):
        self.variables = variables
        assert len(low)==len(high), "length of low and high must be equal"
        self.low = np.array(low) if type(low)==list else low
        self.high = np.array(high) if type(high)==list else high
        self.normalized = normalized
        if normalized:
            self.func = lambda self: normalize(func(self).astype("float32"), low, high)
            self.observation_space = Box(-1,1,(len(low,)))
        else:
            self.func = lambda self: func(self).astype("float32")
            self.observation_space = Box(low, high)
        if description == "":
            description = str([var.name for var in variables])
        self.description = description
    
    
    def __str__(self) -> str:
        return self.__dict__()
    
    
    def __repr__(self) -> str:
        return self.__dict__()
    
    
default_obs = Observation(
    [pose,velocity_vector,cable_forces],
    lambda self: np.hstack(self.pose.ravel(), self.pose_dot.ravel(), self.f.ravel()),
    low = [-5,0,-100,-100,150,150,150],
    high = [5,5,100,100,2500,2500,2500],
    normalized = True,
    description = "Observe [pose, velocities, forces]")
    
def obs_func_1(self):
    """
    Observe [pose, velocities, accelerations, forces]
    """
    pose = normalize(self.pose.ravel(), self.axes[0,:], self.axes[1,:])
    velocities = normalize(self.pose_dot.ravel(),-100,100)
    accelerations = normalize(self.pose_dot_dot.ravel(),-100,100)
    forces = normalize(self.f, self.f_min, self.f_max)
    return np.hstack(pose, velocities, accelerations, forces).astype("float32")

def obs_func_2(self):
    """
    Observe [pose, velocities, forces, steps]
    """
    pose = normalize(self.pose.ravel(), self.axes[0,:], self.axes[1,:])
    velocities = normalize(self.pose_dot.ravel(),-100,100)
    forces = normalize(self.f, self.f_min, self.f_max)
    return np.hstack(pose, velocities, forces, self.steps).astype("float32")
    
def obs_func_3(self):
    """
    Observe [pose, velocities, forces, I]
    """
    pose = normalize(self.pose.ravel(), self.axes[0,:], self.axes[1,:])
    velocities = normalize(self.pose_dot.ravel(),-100,100)
    forces = normalize(self.f, self.f_min, self.f_max)
    I = self.scale(np.clip(self.I/100,0,10),0,10)
    return np.hstack(pose, velocities, forces, I).astype("float32")    
# %%
