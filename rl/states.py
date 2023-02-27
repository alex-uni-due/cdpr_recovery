#%%
import numpy as np
from collections import namedtuple
#%%
Property = namedtuple("Property", ["name", "symbol", "description","dtype"])
StateVariable = namedtuple("StateVariable", ["name","symbol", "unit", "description", "dtype","size"])
DerivedVariable = namedtuple("DerivedVariable", ["name", "symbol", "unit", "description","dtype", "access","size"])

#%% CDPR properties
translational_dof = Property("translational_dof","trans_dof", "translational degrees of freedom",int)
rotational_dof = Property("rotational_dof","trans_dof", "rotational degrees of freedom",int)
degrees_of_freedom = Property("degrees_of_freedom", "n", "degrees of freedom", int)
number_of_cables = Property("number_of_cables", "m", "number of cables", int)
frame_attachment_points = Property("frame_attachment_points","B", "frame attachment points",np.ndarray)
platform_attachment_points = Property("platform_attachment_points","P", "platform attachment points",np.ndarray)
axes = Property("axes","axes", "axes",list)
translational_axes = Property("translational_axes","trans_axes", "translational axes",list)
rotational_axes = Property("rotational_axes","rot_axes", "rotational axes",list)
goal_pose = Property("goal_pose","goal_pose", "platform goal pose",np.ndarray)
goal_position = Property("goal_position","goal_position", "platform goal position", np.ndarray)
goal_orientation = Property("goal_orientation","goal_orientation", "platform goal orientation", np.ndarray)

#%% CDPR state variables
pose = StateVariable("pose","pose", "[m|rad]", "platform pose", np.ndarray, "n")
spatial_velocity = StateVariable("spatial_velocity","pose_dot", "[m/s|rad/s]", "platform pose velocities", np.ndarray, "n")
spatial_accelaration = StateVariable("spatial_accelaration","pose_dot_dot", "[m/s**2|rad/s**2]", "platform pose accelerations", np.ndarray, "n")

cable_lengths = StateVariable("cable_lengths","l","[m]","cable lengths vector", np.ndarray, "m")
cable_forces = StateVariable("cable_forces","f", "[N]", "applied cable force", np.ndarray, "m")
cable_force_gradients = StateVariable("cable_force_gradients","delta_f", "[N]", "gradient of applied cable forces", np.ndarray, "m")

wrench = StateVariable("wrench","w", "[N|N/m]", "platform wrench", np.ndarray, "n")
external_wrench = StateVariable("external_wrench","w_e", "[N|N/m]", "platform wrench", np.ndarray, "n")

steps = StateVariable("steps","steps", "", "number of timesteps", int, 1)
# jacobian = StateVariable("jacobian","A", "[]", "jacobian matrix", np.ndarray, "(self.n,self.m)")
# rotation_matrix = StateVariable("rotation_matrix", "R", "[]", "rotatation matrix", np.ndarray, "(self.trans_dot, self.trans_dof)")
# mass_matrix = StateVariable("mass_matrix", "M", "[]", np.ndarray, "(self.trans_dot, self.trans_dof))")
#%% CDPR derived state variables
position = DerivedVariable("position","r_p", "m", "platform r_p", np.ndarray,"self.pose[:self.trans_dof]","trans_dof")
orientation = DerivedVariable("orientation","phi","rad","orientation of platform", np.ndarray,"self.pose[-self.rot_dof:]","rot_dof")

velocity_vector = DerivedVariable("velocity_vector","r_p_dot", "[m/s]", "platform velocity vector", np.ndarray,"self.pose_dot[:self.trans_dof]","trans_dof")
acceleration_vector = DerivedVariable("acceleration_vector","r_p_dot_dot", "[m/s**2]", "platform acceleration vector", np.ndarray,"self.pose_dot_dot[:self.trans_dof]","trans_dof")
angular_velocity_vector = DerivedVariable("angular_velocity_vector", "Phi_dot", "[rad/s]", "platform angular velocity vector", np.ndarray,"self.pose_dot_dot[-self.rof_dof:]","rot_dof")
angular_acceleration_vector = DerivedVariable("angular_acceleration_vector", "Phi_dot_dot", "[rad/s**2]", "platform angular acceleration vector", np.ndarray,"self.pose_dot_dot[-self.rot_dof:]","rot_dof")

velocity = DerivedVariable("velocity","v", "[m/s]", "platform velocity", float,"norm(self.r_p_dot)",1)
acceleration = DerivedVariable("acceleration","a", "[m/s**2]", "platform acceleration", float,"norm(self.r_p_dot_dot)",1)
angular_velocity = DerivedVariable("angular_velocity","omega", "[rad/s]", "platform angular velocity", float,"norm(self.angular_velocity_vector)",1)
angular_acceleration = DerivedVariable("angular_acceleration","omega_dot", "[rad/s]", "platform angular acceleration", float,"norm(self.angular_acceleration_vector)",1)

pose_error = DerivedVariable("pose_error","e", "[m|rad]", "platform pose error towards goal pose", np.ndarray, "self.goal_pose-self.pose","n")
position_error = DerivedVariable("position_error","e_r_p", "[m]", "platform position error towards goal position", np.ndarray, "self.goal_position-self.r_p","trans_dof")
orietation_error = DerivedVariable("orietation_error","e_phi", "[m]", "platform orietation error towards goal orientation", np.ndarray, "self.goal_orietation-self.Phi","rot_dof")
position_error_norm = DerivedVariable("position_error_norm","e_r_p_abs", "[m]", "euclidean norm of platform position error towards goal position", float, "norm(position_error)",1)
orietation_error_norm = DerivedVariable("orietation_error_norm","e_phi_abs", "[m]", "euclidean norm of platform orietation error towards goal orientation", float, "norm(orientation_error)",1)

# %%
default_state_variables = [pose, spatial_velocity, spatial_accelaration, cable_forces, cable_force_gradients]