#%%
import os
from numpy import array
wdir = os.getcwd()
#%%
from cdpr.utils import *
from cdpr.cdpr import *
from cdpr.configurations import *
#%%
# 2D Coordinates
coordinate_system_2D = CoordinateSystem([TranslationalAxisY(-5,5),TranslationalAxisZ(0,5)])
frame_corners_2D = array([[-5, -5, 5, 5],
                          [0, 5, 5, 0]])

frame_attachment_points_2D = array([[-5, -5, 5, 5],
                                    [0.5, 4.5, 0.5, 4.5]])

frame_2D = Frame(frame_corners_2D, frame_attachment_points_2D)

# 2D point mass parameters
platform_point_mass = PointMass(mass=32.5)
wrench_point_mass_2D = array([[0],[-platform_point_mass.mass*9.81]])
cables_point_mass_2D = Cables({0:None, 1:None, 2:None, 3:None},
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
platform_body_corners_2D = array([[-0.45, -0.45, 0.45, 0.45],
                                  [-0.235, 0.235, 0.235, -0.235]])
platform_body_attachment_points_2D = array([[-0.45, -0.45, 0.45, 0.45],
                                            [-0.235, 0.235, -0.235, 0.235]])
platform_body_2D = Body2D(mass=32.5,
                          corners = platform_body_corners_2D,
                          attachment_points= platform_body_attachment_points_2D,
                          inertia=10, #around x-axis [kg*m**2],
                          **{"width": 0.9, "height": 0.47}
                          )

# 2D body parameters
coordinate_system_2D_rotational = CoordinateSystem([TranslationalAxisY(-5,5),TranslationalAxisZ(0,5),RotationalAxisX()])
wrench_body_2D = array([[0],[-platform_point_mass.mass*9.81],[0]])
cables_body_2D = Cables({0:0, 1:1, 2:2, 3:3},
                          f_min = 150,
                          f_max=2500)

cables_body_2D_crossed = Cables({0:1, 1:0, 2:3, 3:2},
                          f_min = 150,
                          f_max=2500)

cablar1r2t = CDPR1R2T(coordinate_system_2D_rotational,
                      frame_2D,
                      platform_body_2D,
                      cables_body_2D,
                      wrench_body_2D)

cablar1r2t_crossed = CDPR1R2T(coordinate_system_2D_rotational,
                      frame_2D,
                      platform_body_2D,
                      cables_body_2D_crossed,
                      wrench_body_2D)
# %%