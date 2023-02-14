import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

_linestyles = [
    ('solid', 'solid'),      # Same as (0, ()) or '-'
    ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
    ('dashed', 'dashed'),    # Same as '--'
    ('dashdot', 'dashdot'),  # Same as '-.'
    ('dotted',                (0, (1, 1))),
    ('long dash with offset', (5, (10, 3))),
    ('densely dotted',        (0, (1, 1))),
    ('loosely dashed',        (0, (5, 10))),
    ('dashed',                (0, (5, 5))),
    ('densely dashed',        (0, (5, 1))),
    ('loosely dashdotted',    (0, (3, 10, 1, 10))),
    ('dashdotted',            (0, (3, 5, 1, 5))),
    ('densely dashdotted',    (0, (3, 1, 1, 1))),
    ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

def round_up(x):
    digits = len(str(x))
    rounded = np.round(float(x),-digits)
    if rounded<x:
        rounded += 10**digits
    return rounded, digits

def visualize_tests_2D(cdpr, successes, failures, timeout):
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    axes_names = cdpr.axes_names
    x_min,x_max,y_min,y_max = cdpr.axes.ravel()
    b = cdpr.b
    b_left = b[:,b[0,:]==x_min]
    b_right = b[:,b[0,:]==x_max] 
    title = "Bergung bei verschiedenen Startpositionen"
    ax.set_title(title, fontsize = 16)
    ax.scatter(b_left[0,:],b_left[1,:], s=60, c = "k", marker=9)
    ax.scatter(b_right[0,:],b_right[1,:], s=60, c = "k", marker=8)
    ax.scatter(successes[:,0],successes[:,1], s=60, c = "g", marker="o", label="Erfolg")
    ax.scatter(failures[:,0],failures[:,1], s=40, c = "y", marker="D", label="nicht gestoppt")
    ax.scatter(timeout[:,0],timeout[:,1], s=70, c = "r", marker="x",label="Kollision")
    # ax.legend(fontsize = 14, loc= 'upper center', ncol= 3)
    ax.legend(bbox_to_anchor=(0.5,1.15), fontsize = 14, loc="upper center", ncol= 3)
    ax.set_xticks(np.arange(x_min, x_max+1, 1.0))
    ax.set_yticks(np.arange(y_min, y_max+1, 1.0))
    ax.set_xlabel(axes_names[0], fontsize = 14)
    ax.set_ylabel(axes_names[1], fontsize = 14)
    ax.set_xticklabels(ax.get_xticks(), fontsize = 12)
    ax.set_yticklabels(ax.get_yticks(), fontsize = 12)
    return fig

def visualize_movement(cdpr, Ts, positions):
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    axes_names = cdpr.axes_names
    x_min,x_max,y_min,y_max = cdpr.axes.ravel()
    b = cdpr.b
    b_left = b[:,b[0,:]==x_min]
    b_right = b[:,b[0,:]==x_max] 
    ax.scatter(b_left[0,:],b_left[1,:], s=60, c = "k", marker=9)
    ax.scatter(b_right[0,:],b_right[1,:], s=60, c = "k", marker=8)
    start, =ax.plot(positions[0,0], positions[1,0],"ro", ms=5, label="start")
    end, = ax.plot(positions[0,-1], positions[1,-1],"gx", label="end", ms=15,mew=2)
    ax.plot(positions)
    ax.set_title(f"Bewegung des Endeffektors", fontsize = 14)
    ax.grid(1)
    ax.legend(bbox_to_anchor=(0.5,1.15), fontsize = 14, loc="upper center", ncol= 3)
    ax.set_xticks(np.arange(x_min, x_max+1, 1.0))
    ax.set_yticks(np.arange(y_min, y_max+1, 1.0))
    ax.set_xlabel(axes_names[0], fontsize = 14)
    ax.set_ylabel(axes_names[1], fontsize = 14)
    ax.set_xticklabels(ax.get_xticks(), fontsize = 12)
    ax.set_yticklabels(ax.get_yticks(), fontsize = 12)
    return fig

def visualize_forces(cdpr, pose, Ts, forces):
    fig, ax = plt.subplots(constrained_layout=True, figsize = (7,3.5))
    m = cdpr.m
    cables_idx = cdpr.cables_idx
    y_max, digits = round_up(cdpr.f_max)
    if cables_idx is None:
        cables_idx = list(range(len(m)))
    for i in range(len(m)):
        ax.plot(forces[1][0:],ls=_linestyles[i], label=f"$f_{cables_idx[i]+1}$")
    ax.set_title(f"Seilkräfte Startpos. = {pose}", fontsize = 14)
    ax.grid(1)
    ax.set_xticks(ax.get_xticks())
    ax.margins(0.02)
    ax.set_xticklabels(np.round(ax.get_xticks()*Ts,3), fontsize = 12)
    ax.set_yticks(np.arange(0, y_max, 10**digits))
    ax.set_yticklabels(ax.get_yticks(), fontsize = 12)
    ax.set_ylim(-10**(digits-1), y_max) 
    ax.set_xlabel(f"t [s]", fontsize = 12)
    ax.set_ylabel("f [N]", fontsize = 12)
    ax.legend(bbox_to_anchor=(1.2,0.5), fontsize = 12, loc="center right")
    return fig
    
def visualize_velocities(cdpr, pose, Ts, velocities, angular_velocities=None):
    fig, ax = plt.subplots(constrained_layout=True, figsize = (7,3.5))
    y_max1, digits1 = round_up(np.max(velocities))
    trans_axes = cdpr.trans_axes
    for i in range(cdpr.trans_dof):
        ax.plot(velocities[1][0:],ls=_linestyles[i], label=f"$f\dot {trans_axes[i]}$")
        
    if angular_velocities is not None:
        ax1 = ax.twinx()
        y_max2, digits2 = round_up(np.max(angular_velocities))
        rot_axes = cdpr.rot_axes
        for j in range(cdpr.rot_dof):
            ax1.plot(angular_velocities[1][0:],ls=_linestyles[j+i], label=f"$f\dot {rot_axes[j].rotation_symbol}$")
    ax.set_title(f"Geschwindigkeiten = {pose}", fontsize = 14)
    ax.grid(1)
    ax.set_xticks(ax.get_xticks())
    ax.margins(0.02)
    ax.set_xticklabels(np.round(ax.get_xticks()*Ts,3), fontsize = 12)
    ax.set_yticks(np.arange(0, y_max1, 10**digits1))
    ax.set_yticklabels(ax.get_yticks(), fontsize = 12)
    ax.set_ylim(-10**(digits1-1), y_max1)
    ax1.set_ylim(-10**(digits2-1), y_max2)  
    ax.set_xlabel(f"t [s]", fontsize = 12)
    ax.set_ylabel("v [m/s]", fontsize = 12)
    ax1.set_ylabel("$\dot \phi$ [rad/s]$", fontsize = 12)
    ax.legend(bbox_to_anchor=(1.2,0.5), fontsize = 12, loc="center right")
    return fig
    
def visualize_poses(cdpr, pose, Ts, poses):
    fig, ax = plt.subplots(constrained_layout=True, figsize = (7,3.5))
    m = cdpr.m
    cables_idx = cdpr.cables_idx
    y_max, digits = round_up(cdpr.f_max)
    if cables_idx is None:
        cables_idx = list(range(len(m)))
    for i in range(len(m)):
        ax.plot(poses[1][0:],ls=_linestyles[i], label=f"$f_{cables_idx[i]+1}$")
    ax.set_title(f"Seilkräfte Startpos. = {pose}", fontsize = 14)
    ax.grid(1)
    ax.set_xticks(ax.get_xticks())
    ax.margins(0.02)
    ax.set_xticklabels(np.round(ax.get_xticks()*Ts,3), fontsize = 12)
    ax.set_yticks(np.arange(0, y_max, 10**digits))
    ax.set_yticklabels(ax.get_yticks(), fontsize = 12)
    ax.set_ylim(-10**(digits-1), y_max) 
    ax.set_xlabel(f"t [s]", fontsize = 12)
    ax.set_ylabel("f [N]", fontsize = 12)
    ax.legend(bbox_to_anchor=(1.2,0.5), fontsize = 12, loc="center right")
    return fig
    