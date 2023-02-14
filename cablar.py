#%%
import os
from numpy import array
wdir = os.getcwd()
#%%
from helper.utils import *
from cdpr.cdpr import *
from cdpr.configurations import *
#%%
# 2D Coordinates
coordinate_system_2D = CoordinateSystem([TranslationalAxisY(-5,5),TranslationalAxisZ(0,5)])
frame_2D = Frame(array([[-5, -5, 5, 5],
                        [0.5, 4.5, 0.5, 4.5]]))


# 2D point mass parameters
platform_point_mass = PointMass(mass=32.5)
wrench_point_mass_2D = array([[0],[-platform_point_mass.mass*9.81]])
cables_point_mass_2D = Cables({1:None, 2:None, 3:None, 4:None},
                          f_min = 150,
                          f_max = 2500)

cablar2t = CDPR2TPointMass(coordinate_system_2D,
                           frame_2D,platform_point_mass,
                           cables_point_mass_2D,
                           wrench_point_mass_2D
                           )
#%%
# 3D point mass parameters
wrench_point_mass_3D = array([[0],[0],[-platform_point_mass.mass*9.81]])

platform_body_2D = Body2D(mass=32.5,
                          attachment_points=array([[-0.45, -0.45, 0.45, 0.45],
                                                   [-0.235, 0.235, -0.235, 0.235]]),
                          inertia=10 #around x-axis [kg*m**2]
                          )

# 2D body parameters
coordinate_system_2D_rotational = CoordinateSystem([TranslationalAxisY(-5,5),TranslationalAxisZ(0,5),RotationalAxisX()])
wrench_body_2D = array([[0],[-platform_point_mass.mass*9.81],[0]])
cables_body_2D = Cables({1:2, 2:1, 3:4, 4:3},
                          f_min = 150,
                          f_max=2500)

cables_body_2D_crossed = Cables({1:1, 2:2, 3:3, 4:4},
                          f_min = 150,
                          f_max=2500)

cablar1r2t = CDPR1R2T(coordinate_system_2D_rotational,
                      frame_2D,
                      platform_body_2D,
                      cables_body_2D,
                      wrench_body_2D)
# %%