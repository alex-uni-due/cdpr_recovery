#%%
import numpy as np
from numpy.linalg import norm
from typing import Union, List, Dict#, Sequence
from abc import ABC, abstractmethod
#%% CDPR Component Classes
IndexMapping = Dict[int, int]
NamesList = List[str]

class TranslationalAxis:
    def __init__(self, name:str, min:float, max:float) -> None:
        """An axis to consider for transltional movement

        Parameters
        ----------
        name : str
            Axis name
        min : float
            Minumum axis value
        max : float
            Maximum axis value
        """
        self.name = name
        self.min = min
        self.max = max

class RotationalAxis:
    def __init__(self, name: str, rotation_symbol: str) -> None:
        """An axis for rational movement.

        Parameters
        ----------
        name : str
            Axis name
        rotation_symbol : str
            Symbol to use rotation relative to the axis
        """
        self.name = name
        self.rotation_symbol = rotation_symbol

class CoordinateSystem:
    def __init__(self, 
                 axes:List[Union[TranslationalAxis,RotationalAxis]]):
        """A coordinate system described as a list of axes that are consindered 
        for translational and/or rotational movement. 
        
        Parameters
        ----------
        axes : List[Union[TranslationalAxis,RotationalAxis]]
            A list of axes   
        """
        self.axes = axes
        
class Cables:
    def __init__(self, 
                 index_mapping: IndexMapping,
                 l_min: float = 0., 
                 l_max: float = np.inf,
                 f_min: float = 0.,
                 f_max: float = np.inf
                 ) -> None:
        """Class for specifying cables properties and how a frame and a platform are connected

        Parameters
        ----------
        index_mapping : IndexMapping
            Mapping between indecies of frame attachment points and platform attachment points.
            If the platfrom is a point mass the frame attachment points should map to None.
        l_min : Union[float,None], optional
            Minimum cable length in [m], by default 0
        l_max : Union[float,None], optional
            Maximum cable length in [m], by default  np.inf
        f_min : float, optional
            Minimum cable force/tension in [N], by default 0
        f_max : float, optional
            Maximum cable force/tension in [N], by default  np.inf
        """
        self.index_mapping = index_mapping
        self.l_min = l_min 
        self.l_max = l_max 
        self.f_min = f_min 
        self.f_max = f_max
        
class Platform:
    def __init__(self,
                 mass: float,
                 attachment_points: Union[np.ndarray,None],
                 inertia: Union[np.ndarray,float, None]) -> None:
        """A CDPR platform

        Parameters
        ----------
        mass : float
            Mass of the paltform in [kg]
       attachment_points : Union[np.ndarray,None]
            Platform cable attachment points. If None the platform is considered to be a pointmass.
        inertia : Union[np.ndarray,float, None]
            Inertia of the platform. If None the platform is considered to be a pointmass.
        """
        self.mass = mass
        if attachment_points is None:
            assert inertia is None, f"The intertia cannot be {inertia} if P is None"
        if inertia is None:
            assert attachment_points is None, f"The attachment points cannot be {attachment_points} if the inertia is None"
        self.attachment_points = attachment_points
        self.inertia = inertia
        
class Frame:
    def __init__(self,attachment_points: np.ndarray) -> None:
        """A CDPR Frame

        Parameters
        ----------
        attachment_points : np.ndarray
            Frame cable attachment points
        """
        self.attachment_points = attachment_points

#%% CDPR Special Components
class PointMass(Platform):
    def __init__(self, mass: float) -> None:
        super().__init__(mass, attachment_points=None, inertia=None)

class Body2D(Platform):
    def __init__(self, mass: float, attachment_points: np.ndarray, inertia: float) -> None:
        super().__init__(mass, attachment_points, inertia)
        
class Body3D(Platform):
    def __init__(self, mass: float, attachment_points: np.ndarray, inertia: np.ndarray) -> None:
        super().__init__(mass, attachment_points, inertia)
        
class RotationalAxisX(RotationalAxis):
    def __init__(self) -> None:
        super().__init__(name="x", rotation_symbol="psi")

class RotationalAxisY(RotationalAxis):
    def __init__(self) -> None:
        super().__init__(name="y", rotation_symbol="theta")

class RotationalAxisZ(RotationalAxis):
    def __init__(self) -> None:
        super().__init__(name="z", rotation_symbol="psi")

class TranslationalAxisX(TranslationalAxis):
    def __init__(self, min:float, max:float) -> None:
        super().__init__(name="x", min=min, max=max)
        
class TranslationalAxisY(TranslationalAxis):
    def __init__(self, min:float, max:float) -> None:
        super().__init__(name="y", min=min, max=max)
        
class TranslationalAxisZ(TranslationalAxis):
    def __init__(self, min:float, max:float) -> None:
        super().__init__(name="z", min=min, max=max)
#%% CDPR Base Class
class CDPR(ABC):
    def __init__(self,
                 coordinate_system: CoordinateSystem,
                 frame: Frame,
                 platform: Platform,
                 cables: Cables,
                 trans_dof: int,
                 rot_dof: int,
                 w_e: np.ndarray,
                 w_min: Union[float, None] = None, 
                 w_max: Union[float, None] = None,
                 **kwargs
                ) -> None:
        """Base Class for creating CDPR models.

        Parameters
        ----------
        coordinate_system : CoordinateSystem
            Cartesian coordinate system specifying translational and rotational Axes.
        frame : Frame
            Frame of CDPR
        platform : Platform
            Paltform of CDPR
        cables : Cables
            Cables of CDPR
        trans_dof : int
            Translational degrees of freedom
        rot_dof : int
            Rotational degrees of freedom
        w_e : numpy.ndarray
            External platform wrench
        w_min : Union[float, None], optional
            Minimum platform wrench, by default None
        w_max : Union[float, None], optional
            Maximum platform wrench, by default None
        """
        # Parameters of the CDPR
        self.g = 9.81 # gravitational constant
        self.coordinate_system = coordinate_system
        self.frame = frame
        self.platform = platform
        self.B = frame.attachment_points
        self.P = platform.attachment_points
        self.cables = cables.index_mapping
        self.default_cables = list(cables.index_mapping.keys())
        self.m_P = platform.mass 
        self.inertia_P = platform.inertia
        self.w_e = w_e
        self.w_min = w_min if w_min else 0.
        self.w_max = w_max if w_max else np.inf
        self.f_min = cables.f_min
        self.f_max = cables.f_max
        self.f_m = (self.f_max + self.f_min)/2 # Medium cable force
        self.l_min = cables.l_min
        self.l_max = cables.l_max

        self.axes_names = []
        self.rot_axes = []
        self.trans_axes = []
        for axis in coordinate_system.axes: 
            if axis.name not in self.axes_names:
                self.axes_names.append(axis.name) 
            if isinstance(axis, RotationalAxis):
                self.rot_axes.append(axis.name)
            elif isinstance(axis, TranslationalAxis):
                self.trans_axes.append(axis.name)
                setattr(self, f"{axis.name}_min", axis.min)
                setattr(self, f"{axis.name}_max", axis.max)
                setattr(self, f"{axis.name}_m", (axis.max+axis.min)/2)
                
        self.trans_dof = trans_dof
        assert len(self.trans_axes)==trans_dof, "The number of translational axes must be equal to the number of translational degrees of freedom"
        
        self.rot_dof = rot_dof
        assert len(self.rot_axes)==rot_dof, "The number of rotational axes must be equal to the number of rotational degrees of freedom"

        if self.P is None or self.P.size == 0:
           assert rot_dof == 0, "rot_dof cannot be greater than 0 if the platform is a pointmass"

        self.n = self.trans_dof + self.rot_dof # Degrees of freedom (dof)
        self.m = len(self.cables) # number of cables
        self.r = self.m - self.n # redundancy
        self.cdpr_type = f"{self.rot_dof}R{self.trans_dof}T"
        self.axes = np.zeros((self.trans_dof,2))
        if trans_dof>1:
            self.axes[0,0] = getattr(self, self.trans_axes[0]+"_min")
            self.axes[1,0] = getattr(self, self.trans_axes[1]+"_min")
            self.axes[0,1]= getattr(self, self.trans_axes[0]+"_max")
            self.axes[1,1] = getattr(self, self.trans_axes[1]+"_max")
            if trans_dof == 3:
                self.axes[2,0] = getattr(self, self.trans_axes[2]+"_min")
                self.axes[2,1] = getattr(self, self.trans_axes[2]+"_max")
            self.e_max = norm(self.axes[:,1]-self.axes[:,0])
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @property
    def platform_attachment_points(self):
        return self.P
    
    @property
    def frame_attachment_points(self):
        return self.B
    
    @property
    def platform_mass(self):
        return self.m_P
    
    @property
    def platform_inertia(self):
        return self.inertia_P
    
    @property
    def wrench_external(self):
        return self.w_e
    
    @property
    def degrees_of_freedom(self):
        return self.n
        
    @property
    def number_of_cables(self):
        return self.m
    
    @abstractmethod
    def calc_jacobian(self,pose,m,b,p):
        """Function for calculating the jacobian matrix of specified configuration
        
        Returns
        -------
        np.ndarray
            jacobian matrix
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_jacobian(self):
        """Function for returning the jacobian matrix of the current configuration.
        
        Returns
        -------
        np.ndarray
            jacobian matrix
        """
        raise NotImplementedError
    
    @abstractmethod
    def calc_rotmat(self, pose):
        """Function for calculating the rotation matrix of specified configuration
        
        Returns
        -------
        np.ndarray
            rotation matrix .
        """
        raise NotImplementedError
        
    @abstractmethod
    def get_rotmat(self):
        """Function for returning the rotation matrix of the current configuration.
        
        Returns
        -------
        np.ndarray
            rotation matrix 
        """
        raise NotImplementedError
    
        
    @abstractmethod
    def calc_mass_matrix(self,m_P):
        raise NotImplementedError
    
    @abstractmethod
    def get_mass_matrix(self):
        raise NotImplementedError
    
    @abstractmethod
    def calc_mass_matrix_inverse(self,m_P):
        raise NotImplementedError
       
    @abstractmethod
    def get_mass_matrix_inverse(self):
        raise NotImplementedError
    
    @abstractmethod
    def calc_ode(self, M_inv:np.ndarray,  A:np.ndarray, f:np.ndarray,  w_e:np.ndarray):
        """Function for calculating ode pose_dot_dot = f(M_inv,A,f,w_e)
        
        Returns
        -------
        np.ndarray
            pose_dot_dot 
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_ode(self):
        """Function for returning current solution of ode pose_dot_dot = f(M_inv,A,f,w_e)
        
        Returns
        -------
        np.ndarray
            pose_dot_dot 
        """
        raise NotImplementedError
