from typing import Callable#, Sequence
from environment.states import *
from inspect import getfullargspec

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
        self.params = getfullargspec(negative_velocity).args
        if description == "":
            description = func.__doc__
        self.description = description
    def __str__(self) -> str:
        return str(self.__dict__)
    def __repr__(self) -> str:
        return str(self.__dict__)
    
#%% Rew Funcs

## Velocity Rewards
def negative_velocity(env, c1:float, c2:float):
    """reward = 1-(c1*env.v**c2)"""
    reward = 1-(c1*env.v**c2)
    return reward
neg_v_reward = Reward(negative_velocity)

def positive_velocity(env, c1:float, c2:float):
    """reward = 1-(c1*env.v**c2)"""
    reward = 1-(c1*env.v**c2)
    return reward
pos_v_reward = Reward(positive_velocity)

def negative_exp_velocity(env, c1:float):
    """reward = 1-np.exp(c1*env.v)"""
    reward = 1-np.exp(c1*env.v)
    return reward
neg_exp_v_reward = Reward(negative_velocity)

def positive_exp_velocity(env, c1:float, c2:float):
    """reward = np.exp(-c1*env.v)"""
    reward = np.exp(-c1*env.v)
    return reward
pos_exp_v_reward = Reward(positive_velocity)

def negative_velocity_acceleration(env, c1:float, c2:float, c3:float, c4:float):
    """reward = -c1*env.v**c2 -c3*env.a**c4"""
    reward = -c1*env.v**c2 -c3*env.a**c4
    return reward
neg_v_a_reward = Reward(negative_velocity_acceleration)

def negaitve_velocity_success(env, c1:float, c2:float, c3:float):
    """reward = c1 if env.success else -(c2*env.v)**c3"""
    reward = c1 if env.success else -(c2*env.v)**c3
    return reward

neg_v_s_reward = Reward(negaitve_velocity_success)

def negaitve_velocity_failure(env, c1:float, c2:float, c3:float):
    """reward = -c1 if env.failure else -(c2*env.v)**c3"""
    reward = -c1 if env.failure else -(c2*env.v)**c3
    return reward

neg_v_f_reward = Reward(negaitve_velocity_failure)

def negaitve_velocity_success_failure(env, c1:float, c2:float, c3:float, c4:float):
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

neg_v_f_s_reward = Reward(negaitve_velocity_success_failure)

def positive_velocity_success(env, c1:float, c2:float, c3:float):
    """reward = c1 if env.failure else 1-c2*env.v**c3"""
    reward = c1 if env.success else 1-c2*env.v**c3
    return reward

pos_v_s_reward = Reward(positive_velocity_success)

def negaitve_velocity_failure(env, c1:float, c2:float, c3:float):
    """reward = -c1 if env.failure else 1-c2*env.v**c3"""
    reward = -c1 if env.failure else 1-c2*env.v**c3
    return reward

neg_v_f_reward = Reward(negaitve_velocity_failure)


def negaitve_velocity_steps(env, c1:float, c2:float, c3:float):
    """reward = -c1*env.v**c2-c3*env.steps"""
    reward = -c1*env.v**c2-c3*env.steps
    return reward

neg_v_step_reward = Reward(negaitve_velocity_steps)

def negaitve_velocity_integral(env, c1:float, c2:float, c3:float):
    """reward = -(c1*env.v**c2)-c3*(env.I/env.steps)"""
    reward = -(c1*env.v**c2)-c3*(env.I/env.steps)
    return reward

neg_v_I_reward = Reward(negaitve_velocity_integral)

def asymptotic_velocity(env, c1:float):
    """reward = 1-(c1*env.v)/(1+c1*env.v)"""
    reward = 1-(c1*env.v)/(1+c1*env.v)
    return reward

asym_v_reward = Reward(asymptotic_velocity)

## Distance rewards
def negative_distance(env, c1:float, c2:float):
    """reward = -(c1*env.distance)**c2"""
    reward = -(c1*env.distance)**c2
    return reward
neg_d_reward = Reward(negative_distance)

def positive_distance(env, c1:float, c2:float):
    """reward = 1-(c1*env.distance)**c2"""
    reward = 1-(c1*env.distance)**c2
    return reward    
pos_d_reward = Reward(positive_distance)

def negative_exp_distance(env, c1:float):
    """reward = 1-np.exp(c1*env.distance)"""
    reward = 1-np.exp(c1*env.distance)
    return reward
neg_exp_d_reward = Reward(negative_exp_distance)

def positive_exp_distance(env, c1:float):
    """reward = np.exp(-c1*env.distance)"""
    reward = np.exp(-c1*env.distance)
    return reward    
pos_exp_d_reward = Reward(positive_exp_distance)