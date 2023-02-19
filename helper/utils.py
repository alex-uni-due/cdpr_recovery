
import itertools
import numpy as np
from numpy import array
from scipy.optimize import linprog
from qpsolvers import solve_qp
from typing import Union,  Literal, List #, Sequence

def intersect(A:np.ndarray,B:np.ndarray,C:np.ndarray,D:np.ndarray)->bool:
    """Check for intersection of line AB with line CD

    Parameters
    ----------
    A : np.ndarray
        Edge of line AB
    B : np.ndarray
        Edge of line AB
    C : np.ndarray
        Edge of line CD
    D : np.ndarray
        Edge of line CD

    Returns
    -------
    bool
        True if intersection occured
    """
    def ccw(A,B,C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def normalize(var: Union[int, float, np.ndarray], 
              var_min: Union[int, float, np.ndarray],
              var_max: Union[int, float, np.ndarray])->Union[int, float, np.ndarray]:
    """Normalizes a variable within the interval [-1,1]
    
    Parameters
    ----------
    var : Union[int, float, np.ndarray]
        Variable to be normalized
    var_min : Union[int, float, np.ndarray]
        Lower bound of variable
    var_max : Union[int, float, np.ndarray]
        Upper bound of variable

    Returns
    -------
    Union[int, float, np.ndarray]
        Normalized variable
    """
    
    var = (var-(var_max+var_min)/2)/((var_max-var_min)/2)
    return var
    
def rescale(var: Union[int, float, np.ndarray], 
            var_min: Union[int, float, np.ndarray], 
            var_max: Union[int, float, np.ndarray])->Union[int, float, np.ndarray]:
    """Rescales a normalized variable

    Parameters
    ----------
    var : Union[int, float, np.ndarray]
        Normalized variable to be rescaled 
    var_min : Union[int, float, np.ndarray]
        Lower bound of variable
    var_max : Union[int, float, np.ndarray]
        Upper bound of variable

    Returns
    -------
    Union[int, float, np.ndarray]
        Rescaled variable
    """
    var = (var *(var_max-var_min)/2)+((var_max+var_min)/2)
    return var
        
### Linear Programming
def linprog(AT:np.ndarray, w_e:np.ndarray, f_min:float, f_max:float) -> np.ndarray:
    """Linear Programming method for calculating cable forces

    Parameters
    ----------
    AT : np.ndarray
        Structure Matrix
    w_e : np.ndarray
        Platform wrench
    f_min : float
        Minimum cable force
    f_max : float
        Maximum cable force

    Returns
    -------
    np.ndarray
        cable forces
    """
    
    m = AT.shape[1]
    c = array([[1,1,1]]).T
    A_ub = np.vstack((np.eye(m),-1*np.eye(m)))
    b_ub = array([[f_max]*m+ [-f_min]*m]).T
    A_eq = AT
    b_eq = -1*w_e
    res = linprog(c, A_ub, b_ub, A_eq, b_eq)
    if res.success:
        f = res.x
    else:
        f = None
    return f

### Quadratic Programming
def quadprog(AT:np.ndarray, w_e:np.ndarray, f_min:float, f_max:float) -> np.ndarray:
    """Quadratic Programming method for calculating cable forces

    Parameters
    ----------
    AT : np.ndarray
        Strucuture Matrix
    w_e : np.ndarray
        Platform wrench
    f_min : float
        Minimum cable force
    f_max : float
        Maximum cable force

    Returns
    -------
    np.ndarray
        cable forces
    """
    m = AT.shape[1]
    P = np.eye(m)
    q = np.zeros(m)
    G = np.vstack((np.eye(m),-1*np.eye(m)))
    h = array([f_max]*m + [-f_min]*m)
    f = solve_qp(P,q,G,h,AT,-1*w_e.ravel(), solver="quadprog")     
    return f
    
### Closed Form Method
def closed_form_method(AT:np.ndarray, w_e:np.ndarray, f_min:float, f_max:float) -> np.ndarray:
    """Closed Form method for calculating cable forces

    Parameters
    ----------
    AT : np.ndarray
        Structure Matrix
    w_e : np.ndarray
        Platform wrench
    f_min : float
        Minimum cable force
    f_max : float
        Maximum cable force

    Returns
    -------
    np.ndarray
        cable forces
    """
    m = AT.shape[1]
    fm = (f_max + f_min)/2 
    f = (fm-np.linalg.pinv(AT)@(w_e+AT@fm))[:m]
    return f
    
def nearest_corner(AT:np.ndarray, w_e:np.ndarray, f_min:float, f_max:float, p_norm=2 )-> np.ndarray:
    """Nearest Corner method for calculating cable forces when desired wrench is outside the wrench feasible workspace

    Parameters
    ----------
    AT : np.ndarray
        Structure Matrix
    w_e : np.ndarray
        Platform wrench
    f_min : float
        Minimum cable force
    f_max : float
        Maximum cable force
    p_norm : int, optional
        p norm to calculate distance from corners, by default 2

    Returns
    -------
    np.ndarray
        cable forces
    """
    
    
    n,m = AT.shape
    u, s, vh = np.linalg.svd(AT)
    H = (vh[1+1:m,:]).T
    h1=H[:,[0]]
    Ap = np.linalg.pinv(AT)
    f0=-Ap@w_e
    minmax = [f_min,f_max]
    Fcorner = np.zeros((m,2**m))
    All_L=np.zeros((2**m,1))
    F1, F2, F3 = np.meshgrid(*(minmax,minmax,minmax))
    for i in range(2**m):
        Fcorner[:,i] = array((F1.flat[i],F2.flat[i],F3.flat[i]))
        Ftemp = f0 + (((Fcorner[:,[i]] - f0).T@h1/(h1.T@h1))*h1)
        LTemp=np.linalg.norm(Fcorner[:,[i]]-Ftemp)
        All_L[i]=LTemp
    l_sum=np.sum(All_L)
    kw_All_L=np.zeros((2**m,1))
    gew=p_norm
    for i in range(2**m):
        kw_All_L[i] = (1/(All_L[i]/l_sum))**gew
    kw_l_sum=np.sum(kw_All_L)
    f_valid = np.zeros((m,1))
    for i in range(2**m):
        f_valid += Fcorner[:,[i]]*(kw_All_L[i]/kw_l_sum)
    return f_valid
    
def euler_cromer(x:np.ndarray, x_dot:np.ndarray, Ts:float):
    """Function for calculating the state of the next timestep using Euler-Cromer method.

    Parameters
    ----------
    x : np.ndarray
        state
    x_dot : np.ndarray
        state derivative
    Ts : float
        timestep

    Returns
    -------
    np.ndarray
        new state
    """
    x = x + x_dot*Ts
    return x

def create_grid_coordinates(spaces:List[np.ndarray]):
    """Function for meshing coordinate spaces into a grid 
    and creating a list-like array of the grid coordiantes.

    Args:
        spaces (List[np.ndarray]): List of coordinate spaces

    Returns:
        np.ndarray: An array of all coordinates on the meshgrid 
    """
    dim0 = 1
    for space in spaces:
        dim0 = dim0*len(space)
    dim1 = len(spaces)
    grid_coordinates = np.zeros((dim0,dim1))
    for i,coords in enumerate(itertools.product(*spaces)):
        coords = np.array(coords)
        grid_coordinates[i] = coords.ravel()
    return grid_coordinates

# Static Workspace Calculation (WS) on discrete arrays
def calc_static_workspace(
    cdpr,
    cables_idx:list, 
    coordinates:np.ndarray,
    method: Literal["lp","qp","cfm"] = "qp") -> np.ndarray:

    """Method for calculating coordinates of the static workspace

    Parameters
    ----------
    cdpr: CDPR
        An instance of a CDPR Class
    cables_idx : list
        List of remaining cables that will used for WS calculation
    coordinates : int
        coordinates that will be checked
    method : Literal["lp","qp","cfm"], optional
        Method that is used to calculate valid cable forces with, by default "qp"\n
        -"lp"  = Linear Programming\n
        -"qp"  = Quadratic Programming\n
        -"cfm" = Closed Form Method

    Returns
    -------
    np.ndarray
        Coordintes of static workspace
    """
    
    w_e = cdpr.w_e.ravel()
    b = cdpr.B[:,cables_idx]
    try:
        p = cdpr.P[:,cables_idx]
    except TypeError:
        p = None
    n = cdpr.n
    m = len(cables_idx)
    method = {"cfm": closed_form_method, 
              "qp": quadprog, 
              "lp": linprog}[method]
                
    ws = np.full((coordinates.shape),np.nan)
    forces = np.full((coordinates.shape[0],m),np.nan)
    for i,coords in enumerate(coordinates):
        pose = array([coords]).reshape(cdpr.n,1)
        AT = cdpr.calc_structure_matrix(pose,b,p)
        if np.linalg.matrix_rank(AT) == n:
            f = method(AT,w_e,cdpr.f_min,cdpr.f_max)
            if (type(f) != type(None)): #and not(np.any(f<=self.f_min) or np.any(f>=self.f_max)):
                ws[i] = pose.ravel()
                forces[i] = f.ravel()
    ws = ws[ ~np.isnan(ws).any(axis=1),:]
    return ws, forces