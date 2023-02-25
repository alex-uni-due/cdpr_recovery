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
        assert len(self.trans_axes)==2
        assert self.rot_axes == []
        assert self.P is None
        assert self.inertia_P is None
        assert self.B.shape[0]==2
        
    def calc_structure_matrix(self,r_p,b,p=None):
        l_i = b - r_p
        AT = l_i/norm(l_i, axis=0)
        return AT
    
    def calc_rotmat(self, pose):
        return super().calc_rotmat()
    
    def calc_mass_matrix(self):
        M = array([[self.m_P, 0], [0, self.m_P]])
        return M
    
    def calc_mass_matrix_inverse(self):
        M_inv = array([[1 / self.m_P, 0], [0, 1 / self.m_P]])
        return M_inv
    
    def calc_cable_lengths(self, pose,m,b,p):
        r_p = pose[:self.trans_dof]
        q = np.array([norm(b[:, [i]] - r_p) for i in range(m)])
        return q
    
    def calc_ode(self, M_inv, AT, f, w_e):
        r_p_dot_dot = M_inv@(AT@f + w_e)
        return r_p_dot_dot
    
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
        self.P = self.P[:,list(self.cables.index_mapping.values())]
    def calc_structure_matrix(self,pose,b,p):
        R = self.calc_rotmat(pose)
        r_p = pose[:self.trans_dof]
        l_i = b - (r_p+R@p)
        A1 = l_i/norm(l_i, axis=0)
        A2 = np.cross(p,A1,axis=0)
        AT = np.vstack((A1,A2))
        return AT
    
    def calc_rotmat(self, pose):
        phi = pose[-1,0]
        R = np.array([[np.cos(phi), -np.sin(phi)],
                      [np.sin(phi), np.cos(phi)]])
        return R
    
    def calc_mass_matrix(self):
        M = np.array([[ self.m_P, 0, 0], 
                      [0, self.m_P,0],
                      [0, 0, self.inertia_P]])
        return M
    
    def calc_mass_matrix_inverse(self):
        M_inv = np.array([[1/self.m_P, 0, 0], 
                          [0, 1/self.m_P,0],
                          [0, 0, 1/self.inertia_P]])
        return M_inv
    
    def calc_cable_lengths(self, pose,m,b,p):
        R = self.calc_rotmat(pose)
        r_p = pose[:self.trans_dof]
        q = np.array([norm(b[:, [i]] - (r_p+R@p[:,[i]])) for i in range(m)])
        return q
    
    def calc_ode(self, M_inv, AT, f, w_e):
        pose_dot_dot = M_inv@(AT@f + w_e)
        return pose_dot_dot