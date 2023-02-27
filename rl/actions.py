import numpy as np
from inspect import getsource
from typing import Callable, Union    
from gym.spaces import Space, Box
from cdpr.utils import rescale

class Action:
    def __init__(
        self,
        func: Callable,
        action_space: Union[Space,Callable],
        description: str) -> None:
        
        self.func = func
        if callable(action_space):
            self.action_space_func = action_space
            self.action_space = self.get_action_space
        else:
            self.action_space = action_space
        self.description = description 
        
    def get_action_space(self, cdpr):
        return self.action_space_func(cdpr)
    def __str__(self) -> str:
        return str(self.__dict__)
    def __repr__(self) -> str:
        return str(self.__dict__)
    
def calc_cable_forces(self, action):
    f = rescale(action, self.f_min, self.f_max).reshape(self.m,1)
    return f

def calc_platform_wrench(self, action, p_norm = 20):
    f = self.wrench_split(action, p_norm)
    return f

def calc_cable_force_gradients(self, action, delta_f_min = -100, delta_f_max = 100):
    delta_f = rescale(action, delta_f_min, delta_f_max).reshape(self.m,1)
    f = np.clip(self.f + delta_f, self.f_min, self.f_max)
    return f

def cable_action_space(self):
    return Box(-1,1,(self.m,))

def wrench_action_space(self):
    return Box(-1,1,(self.n,))

        
cable_forces_action = Action(
    calc_cable_forces,
    cable_action_space,
    "cable forces"
    )

cable_force_gradients_action = Action(
    calc_cable_force_gradients,
    cable_action_space,
    "cable force gradients"
    )

platform_wrench_action = Action(
    calc_platform_wrench,
    cable_action_space,
    "platform wrench"
    )