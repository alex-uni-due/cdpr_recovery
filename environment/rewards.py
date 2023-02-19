from typing import Callable#, Sequence
from environment.states import *

class Reward:
    def __init__(self, 
                 func:Callable,
                 description: str = "") -> None:
        """Class for providing descriptive rewards

        Parameters
        ----------
        func : Callable
            reward function
        description : str
            Description
        """
        self.func = func
        self.args = func.__code__.co_varnames
        if description == "":
            description = func.__doc__
        self.description = description
    def __str__(self) -> str:
        return str(self.__dict__)
    def __repr__(self) -> str:
        return str(self.__dict__)
    
#%% Rew Funcs
def rew_func_0(env, c1:float, c2:float):
    """reward = -(c1*env.v)**c2"""
    reward = -(c1*env.v)**c2
    return reward

reward_0 = Reward(rew_func_0)

def rew_func_1(env, c1:float, c2:float):
    """reward = 1-(c1*env.v**c2)"""
    reward = 1-(c1*env.v**c2)
    return reward

reward_1 = Reward(rew_func_1)

def rew_func_2(env, c1:float, c2:float, c3:float, c4:float):
    """reward = -c1*env.v**c2 -c3*env.a**c4"""
    reward = -c1*env.v**c2 -c3*env.a**c4
    return reward

reward_2 = Reward(rew_func_2)

def rew_func_3(env, c1:float, c2:float, c3:float):
    """reward = c1 if env.success else -(c2*env.v)**c3"""
    reward = c1 if env.success else -(c2*env.v)**c3
    return reward

reward_3 = Reward(rew_func_3)

def rew_func_4(env, c1:float, c2:float, c3:float):
    """reward = -c1 if env.failure else -(c2*env.v)**c3"""
    reward = -c1 if env.failure else -(c2*env.v)**c3
    return reward

reward_4 = Reward(rew_func_4)

def rew_func_5(env, c1:float, c2:float, c3:float, c4:float):
    """
    if env.success:
        reward = c1
    elif env.failure:
        reward = -c2
    else:
        reward = -(c3*env.v)**c4"""
        
    if env.success:
        reward = c3
    elif env.failure:
        reward = -c4
    else:
        reward = -(c1*env.v)**c2
    return reward

reward_5 = Reward(rew_func_5)

def rew_func_6(env, c1:float, c2:float, c3:float):
    """reward = c1 if env.failure else 1-c2*env.v**c3"""
    reward = c1 if env.success else 1-c2*env.v**c3
    return reward

reward_6 = Reward(rew_func_6)

def rew_func_7(env, c1:float, c2:float, c3:float):
    """reward = -c1 if env.failure else 1-c2*env.v**c3"""
    reward = -c1 if env.failure else 1-c2*env.v**c3
    return reward

reward_7 = Reward(rew_func_7)


def rew_func_8(env, c1:float, c2:float, c3:float):
    """reward = -c1*env.v**c2-c3*env.steps"""
    reward = -c1*env.v**c2-c3*env.steps
    return reward

reward_8 = Reward(rew_func_8)


def rew_func_9(env, c1:float, c2:float, c3:float):
    """reward = -(c1*env.v**c2)-c3*(env.I/env.steps)"""
    reward = -(c1*env.v**c2)-c3*(env.I/env.steps)
    return reward

reward_9 = Reward(rew_func_9)

def rew_func_10(env, c1:float):
    """reward = 1-(c1*env.v)/(1+c1*env.v)"""
    reward = 1-(c1*env.v)/(1+c1*env.v)
    return reward

reward_10 = Reward(rew_func_10)
