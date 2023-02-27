from typing import Callable
from cdpr.utils import intersect
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
            description = f"{success_func.__doc__}\n{failure_func.__doc__}\n{timeout_func.__doc__}"
        self.description = description
    def __str__(self) -> str:
        return str(self.__dict__)
    def __repr__(self) -> str:
        return str(self.__dict__)
    
def task_success(env):
    """success = env.v<=0.01"""
    success = env.v<=0.01 
    return success

def task_success_body2D(env):
    """success = env.v<=0.01"""
    success = env.v<=0.01 and env.omega<=0.01
    return success

def task_collision_failure_pointmass(env):
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

def task_collision_failure_body2D(env):
    corners = env.platform.corners
    R = env.rotation_matrix
    for i in range(corners.shape[1]):
        corner1 = corners[:,[i]]
        corner2 = corners[:,[(i+1)%len(corners)]]
        A = env.r_p + (R@corner1)
        B = env.r_p + (R@corner2)
        for border in env.borders:
            C = border[0]
            D = border[1]
            if intersect(A,B,C,D):
                return True
    return False

def task_body2D_failure(env):
    return task_collision_failure_body2D(env) or task_tilted_failure(env)

def task_timeout(env, n):
    """timeout = env.steps>n"""
    timeout = env.steps>n
    return timeout 
    
stop_pointmass_task = Task(
    task_success,
    task_collision_failure_pointmass,
    task_timeout)

stop_body2D_task = Task(
    task_success_body2D,
    task_collision_failure_body2D,
    task_timeout)