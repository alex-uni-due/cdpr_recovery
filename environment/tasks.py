from typing import Callable
from helper.utils import intersect
class Task:
    def __init__(self,
                 success_func:Callable, 
                 failure_func:Callable,
                 timeout_func:Callable,
                 description:str = "") -> None:
        self.is_success = success_func
        self.is_failure = failure_func
        self.is_timeout = timeout_func
        if description == "":
            description 
        self.description = description
    def __str__(self) -> str:
        return str(self.__dict__)
    def __repr__(self) -> str:
        return str(self.__dict__)
    
    
def task_success(env):
    """success = env.v<=0.01"""
    success = env.v<=0.01
    return success

def task_out_of_bounds_failure(env):
    """failure = (
        env.r_p[0,0] >= env.y_max or
        env.r_p[0,0] <= env.y_min or 
        env.r_p[1,0] >= env.z_max or 
        env.r_p[1,0] <= env.z_min)"""
        
    failure =  (
        env.r_p[0,0] >= env.y_max or 
        env.r_p[0,0] <= env.y_min or 
        env.r_p[1,0] >= env.z_max or 
        env.r_p[1,0] <= env.z_min)
    return failure
   
def task_tilted_failure(env):
    """failure = (env.phi<=env.phi_min or env.phi>=env.phi_max)"""
    failure = (env.phi<=env.phi_min or env.phi>=env.phi_max)
    return failure

def task_collision_failure(env):
    for i in range(4):
        A = env.r_p + (env.R@env.P[:,[env.corners[i]]])
        B = env.r_p + (env.R@env.P[:,[env.corners[i+1]]])
        for border in env.borders:
            C = border[0]
            D = border[1]
            failure = intersect(A,B,C,D)
    return failure

def task_timeout(env, n):
    """timeout = env.steps>n"""
    timeout = env.steps>n
    return timeout 