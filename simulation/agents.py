import numpy as np
from stable_baselines3 import SAC, PPO
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from typing import Union, List, Optional, Callable
from qpsolvers import solve_qp
from scipy.optimize import minimize
from scipy.optimize import linprog
from environment.environment import EmergencyStrategyCDPR
from cdpr.utils import normalize, rescale, quadprog, nearest_corner
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum
from torch import nn as nn

class RLAgent:
    sac_sde_config = dict(
            use_sde=True,
            sde_sample_freq = 8,
            use_sde_at_warmup = True,
            learning_rate=7.3e-4,
            learning_starts=10000,
            gamma=0.98,
            buffer_size=300000,
            batch_size=256,
            ent_coef='auto',
            tau=0.02,
            train_freq=(1, "episode"),
            policy_kwargs= dict(log_std_init=-3,
                                net_arch=[400, 300]))

    tqc_sde_config = sac_sde_config
    
    ppo_sde_config = dict(
        max_grad_norm = 0.5,
        gamma = 0.9,
        ent_coef = 0.0,
        use_sde = True,
        sde_sample_freq = 4,
        n_steps = 512,
        learning_rate = 3e-5,
        n_epochs = 20,
        batch_size = 128,
        clip_range = 0.4,
        gae_lambda = 0.9,
        policy_kwargs= dict(log_std_init=-2,
                            activation_fn=nn.ReLU,
                            ortho_init=True,
                            net_arch=[256, 256]))
    
    def __init__(self,
                 env, 
                 algo="TQC",
                 seed=1,
                 sde=False,
                 n_envs=1) -> None:
        
        self.env = env
        self.algo = algo
        self.seed = seed
        self.sde = sde
        self.n_envs = n_envs
        
        self.vec_env = make_vec_env(lambda: env,
                                    seed = seed,
                                    monitor_kwargs=dict(info_keywords=("is_success",)),
                                    n_envs=n_envs)
        
        if sde == True:
            if algo in ["SAC", "TQC"]:
                self.config = self.sac_sde_config
            else:
                self.config = self.ppo_sde_config
                
        self.model = algo('MlpPolicy',
                     self.vec_env, 
                     device='cuda',
                     verbose=0, 
                     **self.config)

        def __getattr__(self,attr):
            return getattr(self.model, attr)

class PotentialFieldAgent:
    def __init__(self,
                 cdpr: EmergencyStrategyCDPR,
                 zeta:float = 500, 
                 eta_B:float = 10000, 
                 eta_W:float = 10000 , 
                 D_p:float = 100,
                 d:float = 3) -> np.ndarray:
            
            """
            Potential field method for calculating a wrench that incorparates attractive and repulsive forces.
            
            Parameters
            ----------
            zeta:float = 500
                Scaling factor of attracting forces
            eta_B:float = 10000, 
                Scaling factor of repulsive forces of the ground
            eta_W:float = 10000,
                Scaling factor of repulsive forces of the walls
            D_p:float = 100
                Dampening factor of virtual dampening force
            d:float = 3
                Influence distance of potential field of the ground
            Returns
            -------
            Wrench vector as np.ndarray
            """
            self.cdpr = cdpr
            self.env = cdpr
            self.zeta = zeta
            self.eta_B = eta_B
            self.eta_W = eta_W
            self.D_p = D_p
            self.d = d
            
    def calc_wrench(self, state):
        r_E = state[:self.cdpr.trans_dof]
        r_G = self.cdpr.goal
        r_p_dot = state[self.cdpr.trans_dof:]
        zeta  = self.zeta
        eta_B = self.eta_B
        eta_W = self.eta_W
        D_p = self.D_p
        d = self.d
        
        rho_f = np.linalg.norm(r_E-r_G)
        F_rep = np.zeros(2)
        if rho_f <=d:
            F_att = -zeta*(r_E-r_G)
        else:
            F_att = -(d*zeta*(r_E-r_G))/rho_f

        # Ground
        if r_E[1] <= r_G[1]:
            delta_rho_B = np.array([0,1])
            F_rep = eta_B*((1/r_E[1]-1/r_G[1])/(r_E[1]**2))*delta_rho_B
    
        # Walls
        for i in range(self.m):
            rho_0_W = B[0,i]-r_G[0]
            rho_W = B[0,i]-r_E[0]
            if  rho_W<= rho_0_W:
                delta_rho_W = (r_E-np.array([B[0,i],r_E[1]]))/np.linalg.norm(r_E-np.array([B[0,i],r_E[1]]))
                F_rep += eta_W*((1/rho_W-1/rho_0_W)/(rho_W**2))*delta_rho_W
            
        self.F_D = -D_p*r_p_dot.ravel()
        self.F_att = F_att
        self.F_rep = F_rep
        self.w_pot = (self.F_att + self.F_rep + self.F_D)
        return self.w_pot
        
    def get_cable_forces(self):
        w_pot = self.calc_wrench(self.cdpr.r_p,self.cdpr.goal, self.cdpr.r_p_dot, self.cdpr.B)
        return self.calc_cable_forces(w_pot)    
    
    ### cable forces with potential field
    def calc_cable_forces(self,w_pot):
        cdpr = self.cdpr
        f = quadprog(cdpr.A, cdpr.w_e-w_pot, cdpr.f_min, cdpr.f_max)
        if f is None:
            f = nearest_corner(self.A, cdpr.w_e-w_pot, cdpr.f_min, cdpr.f_max, self.p_norm)
        return f.reshape(self.m, 1)
    
    def predict(self, obs):
        state = rescale(obs,
                        self.cdpr.observartion.low, 
                        self.cdpr.observartion.high)
        w_pot = self.calc_wrench(state)
        f = self.calc_cable_forces(w_pot)
        action = self.cdpr.get_action(f)
        return action
        
class PotentialFieldAgentOld(PotentialFieldAgent):
    def __init__(self, cdpr: EmergencyStrategyCDPR, 
                 zeta: float = 500, 
                 eta_B: float = 10000, 
                 eta_W: float = 10000, 
                 D_p: float = 100, 
                 d: float = 3) -> np.ndarray:
        
        super().__init__(cdpr, zeta, eta_B, eta_W, D_p, d)
        
    def calc_cable_forces(self,
                          w_pot:np.ndarray,
                          D1:np.ndarray,
                          D2:np.ndarray,
                          normalized:bool = True) -> np.ndarray:
        """Potential field method for calculating cable forces that consider attractive and repulsive forces.

        Parameters
        ----------
        w_pot : np.ndarray
            Wrench vector including forces applied by potential field, by default None
        D1 : np.ndarray
            Weights matrix for slack variables, by default None
        D2 : np.ndarray
            Weights matrix for cable forces, by default None
        normalize : bool, optional
            Flag of normalizing cable forces, by default True

        Returns
        -------
        np.ndarray
            cable forces vector
        """
        
        if w_pot:
            self.w_pot = w_pot
        else:
            self.w_pot = self.pot_field()
            
        D1 = np.eye(self.n)
        D2 = 0.05*np.eye(self.m)
        self.P = np.diag(np.hstack((np.diag(D1),np.diag(D2),1)))
        self.P[self.n:-1,-1]=-np.diag(D2)
        self.q = np.zeros(self.n+self.m+1)

        self.G = np.zeros((2*self.m,self.n+self.m+1))
        self.G[:,self.n:-1] = np.vstack((np.eye(3),-np.eye(3)))

        h = np.array([self.cdpr.f_max]*self.m+[-self.cdpr.f_min]*self.m)
        A = np.vstack((np.hstack((np.eye(2),self.A.T, np.zeros((self.n,1)))),np.array([[0]*(self.n+self.m)+[1]])))
        b = np.hstack([-self.w_pot,(self.cdpr.f_max+self.cdpr.f_min)/2])
        opt = solve_qp(self.P,self.q,self.G,h,A,b)
        f = opt[self.n:self.n+self.m]
        if normalized:
            return self.scale(f.ravel(), self.cdpr.f_min, self.cdpr.f_max)
        else:
            return f
    
    
# Nonlinear Model Predictive Control (NMPC)

class NMPC_Agent:
    def __init__(self,
                 cdpr: EmergencyStrategyCDPR, 
                 Q:np.ndarray = np.diag((1,2.5)),
                 r:float = 1e-7, 
                 method:str ="Powell",
                 normalize:bool = True):
        """
        Nonlinear Model Predictive Control (NMPC) method with with 1 timestep lookahead. 
        Calculates cable tensions that will minimize an objective function.
        The objective is to minimize velocity and therefore kinetic energy of the endeffector.
        
        Parameters
        ----------
        Q:np.ndarray = np.diag((1,2.5))
            Weighting matrix for velocities.
        r:float = 1e-7, 
            Weighting factor for cable forces
        method:str OneOf["Nelder-Mead","Powell","L-BFGS-B","TNC","SLSQP"] default: "Powell"
            Method that is used to minimize the objective funtion.
        Returns
        -------
        Force vector as np.ndarray
        """
        self.cdpr = cdpr
        self.Q = Q
        self.r = r
        self.method = method
        self.normalize = normalize
        
    def calc_cable_forces(self):
        cdpr = self.cdpr
        ode = cdpr.get_ode
        Q = self.Q
        r = self.r
        f = cdpr.f
        f_prev = cdpr.f_prev
        Ts = cdpr.Ts
        r_p_dot = cdpr.r_p_dot
        J = lambda f: (((ode()*cdpr.Ts+cdpr.r_p_dot).T@ Q @ (ode())*cdpr.Ts+cdpr.r_p_dot)
                        + r*(f-f_prev).T @ (f-f_prev))[0,0]
        f = minimize(J, 
                     f_prev, 
                     bounds = ((cdpr.f_min, cdpr.f_max),(cdpr.f_min, cdpr.f_max),(cdpr.f_min, cdpr.f_max)), 
                     method=self.method).x
        
        if self.scale:
            return normalize(f.ravel(), cdpr.f_min, cdpr.f_max)
        else:
            return f
