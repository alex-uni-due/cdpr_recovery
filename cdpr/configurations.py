#%%
import numpy as np
from cdpr.cdpr import *
from numpy import array
from numpy.linalg import norm
from typing import Union
#%% CDPR Pointmass 2D
class CDPR2TPointMass(CDPR):
    """Class for creating CDPR models with 0 rotational and 2 translational degrees of Freedom.

    Parameters
    ----------
    coordinate_system : CoordinateSystem
        Cartesian coordinate system
    frame : Frame
        Frame of CDPR
    platform : Platform
        Paltform of CDPR
    cables : Cables
        Cables of CDPR
    w_e : numpy.ndarray
        External platform wrench
    w_min : Union[float, None], optional
        Minimum platform wrench, by default None
    w_max : Union[float, None], optional
        Maximum platform wrench, by default None
    """

    def __init__(self,
                 coordinate_system: CoordinateSystem,
                 frame: Frame,
                 platform: Platform,
                 cables: Cables,
                 w_e: np.ndarray,
                 w_min: Union[float, None] = None,
                 w_max: Union[float, None] = None, 
                 **kwargs
                 ) -> None:
        super().__init__(
            coordinate_system=coordinate_system,
            frame=frame,
            platform=platform,
            cables=cables,
            trans_dof=2,
            rot_dof=0,
            w_e=w_e,
            w_min=w_min,
            w_max=w_max,
            **kwargs)
        print(self.rot_axes)
        assert len(self.trans_axes)==2
        assert self.rot_axes == []
        assert self.P is None
        assert self.inertia_P is None
        assert self.B.shape[0]==2
        
    def calc_jacobian(self,r_p,m,b,p=None):
        A = np.zeros((m, self.n))
        for i in range(0, m):
            l_i = b[:, [i]] - r_p
            vi = l_i / norm(l_i)
            A[i, :] = vi.ravel()
        return A
    
    ## Calculate the transposed structure matrix, i.e. the jacobian
    def get_jacobian(self):
        return self.calc_jacobian(self.r_p,self.m,self.b)
    
    def calc_rotmat(self, pose):
        return super().calc_rotmat()
    
    def get_rotmat(self):
        return super().get_rotmat()
    
    def calc_mass_matrix(self,m_P):
        M = array([[self.m_P, 0], [0, self.m_P]])
        return M
    
    def calc_mass_matrix_inverse(self,m_P):
        M_inv = array([[1 / m_P, 0], [0, 1 / m_P]])
        return M_inv
    
    def get_mass_matrix(self):
        M = array([[self.m_P, 0], [0, self.m_P]])
        return M
    
    def get_mass_matrix_inverse(self):
        M_inv = array([[1 / self.m_P, 0], [0, 1 / self.m_P]])
        return M_inv
    
    def calc_ode(self, M_inv, A, f, w_e):
        r_p_dot_dot = M_inv@(A.T@f + w_e)
        return r_p_dot_dot
    
    ## Calculate a state trasition using Euler-Cromer Integration
    def get_ode(self):
        return self.calc_ode(self.get_mass_matrix_inverse(), self.A, self.f, self.w_e())

#%% CDPR Body 2D
class CDPR1R2T(CDPR):
    def __init__(self,
                 coordinate_system: CoordinateSystem,
                 frame: Frame,
                 platform: Platform,
                 cables: Cables,
                 w_e: np.ndarray,
                 w_min: Union[float, None] = None,
                 w_max: Union[float, None] = None, 
                 **kwargs) -> None:
        super().__init__(
            coordinate_system=coordinate_system,
            frame=frame,
            platform=platform,
            cables=cables,
            trans_dof=2,
            rot_dof=1,
            w_e=w_e,
            w_min=w_min,
            w_max=w_max,
            **kwargs)
        assert type(self.B) is np.ndarray and self.B.shape[0]==2, "frame attachment points must contained in an array of shape (2,m)"
        assert type(self.P) is np.ndarray and self.P.shape[0]==2, "platform attachment points must contained in an array of shape (2,m)"
        assert type(self.inertia_P) in [float,int], f"The platform inertia must be a number (in kg*m**2) not {type(self.inertia_P)}"
        assert self.B.shape[0]==2
        # Parameters of the CDPR
    
    def calc_jacobian(self,pose,m,b,p=None):
        A = np.zeros((m, self.n))
        R = self.calc_rotmat(pose)
        r_p = pose[:self.trans_dof]
        for i in range(0, m):
            l_i = b[:, [i]] - (r_p+np.dot(R,p[:,[i]]))
            vi = (l_i / np.linalg.norm(l_i)).ravel()
            A[i, :-1] = vi.ravel()
            A[i, -1] = np.cross(p[:,i],vi)
        return A
    
    ## Calculate the transposed structure matrix, i.e. the jacobian
    def get_jacobian(self):
        return self.calc_jacobian(self.r_p,self.m,self.b)
    
    def calc_rotmat(self, pose):
        phi = pose[-1]
        R = np.array([[np.cos(phi), -np.sin(phi)],
                      [np.sin(phi), np.cos(phi)]])
        return R
    
    def get_rotmat(self):
        return self.calc_rotmat(self.pose)
    
    def calc_mass_matrix(self,m_P,inertia_P):
        M = np.array([[ m_P, 0, 0], 
                      [0, m_P,0],
                      [0, 0, inertia_P]])
        return M
    
    def calc_mass_matrix_inverse(self,m_P,inertia_P):
        M_inv = np.array([[1 / m_P, 0, 0], 
                          [0, 1 / m_P,0],
                          [0, 0, 1/inertia_P]])
        return M_inv
    
    def get_mass_matrix(self):
        return self.calc_mass_matrix(self.m_P,self.inertia_P)
    
    def get_mass_matrix_inverse(self):
        return self.calc_mass_matrix_inverse(self.m_P,self.inertia_P)
    
    def calc_ode(self, M_inv, A, f, w_e):
        pose_dot_dot = M_inv@(A.T@f + w_e)
        return pose_dot_dot
    
    ## Calculate a state trasition using Euler-Cromer Integration
    def get_ode(self):
        return self.calc_ode(self.get_mass_matrix_inverse(), self.A, self.f, self.w_e())