import numpy as np
import matplotlib as mpl
import math
from matplotlib import pyplot as plt
from cdpr.cdpr import CDPR
from cdpr.utils import calc_static_workspace, create_grid_coordinates
from typing import List, Union

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

def _get_digits(x:float):
    digits = len(str(int(x)).replace("-",""))-1
    return digits

def visualize_forces(cdpr:CDPR, pose, Ts, forces):
    fig, ax = plt.subplots(constrained_layout=True, figsize = (6,2))
    m = cdpr.m
    cables_idx = cdpr.cables_idx
    digits = _get_digits(cdpr.f_max)
    y_max = math.ceil(cdpr.f_max/10**digits)*10**digits
    if cables_idx is None:
        cables_idx = list(range(m))
    for i in range(m):
        ax.plot(forces[:,i],ls=_linestyles[i][1], label=f"$f_{cables_idx[i]+1}$")
    ax.set_title(f"Seilkräfte", fontsize = 14) #für Startpos. = {np.round(pose,1)}
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    xlabels = [np.round(i*Ts, len(str(Ts))) for i in xticks]
    ax.set_xticklabels(xlabels, fontsize = 12)
    ax.set_yticks(np.arange(0, y_max, 10**digits/2, dtype="int"))
    ax.set_yticklabels(ax.get_yticks(), fontsize = 12)
    ax.grid(1)
    ax.set_xmargin(0)
    ax.set_xlim(-0.01*len(forces)) 
    ax.set_ylim(-10**(digits-1), y_max) 
    ax.set_xlabel(f"t [s]", fontsize = 12)
    ax.set_ylabel("f [N]", fontsize = 12)
    ax.legend(bbox_to_anchor=(1.2,0.5), fontsize = 12, loc="center right")
    return fig

def visualize_velocities(cdpr:CDPR, pose, Ts, velocities, angular_velocities=None):
    fig, ax = plt.subplots(constrained_layout=True, figsize = (6,2))
    trans_axes = cdpr.trans_axes
    for i in range(cdpr.trans_dof):
        ax.plot(velocities[:,i],ls=_linestyles[i][1], label=f"$\dot {trans_axes[i]}$")
        
    if angular_velocities is not None:
        ax1 = ax.twinx()
        rot_axes = cdpr.rot_axes
        for j in range(cdpr.rot_dof):
            ax1.plot(angular_velocities[:,i],ls=_linestyles[j+i][1], label=f"$\dot {rot_axes[j].rotation_symbol}$")
        ax1.set_yticks(ax1.get_yticks())
        ax1.set_yticklabels(np.round(ax1.get_yticks(),3), fontsize = 12)
        ax1.set_ylabel("$\dot \phi$ [rad/s]$", fontsize = 12) 
    ax.set_title(f"Geschwindigkeiten des Endeffektors", fontsize = 14) #= {np.round(pose,1)}
    ax.set_xticks(ax.get_xticks())
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    xlabels = [np.round(i*Ts, len(str(Ts))) for i in xticks]
    ax.set_xticklabels(xlabels, fontsize = 12)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(np.round(ax.get_yticks(),3), fontsize = 12)
    ax.grid(1)
    ax.set_xmargin(0)
    ax.set_xlim(-0.01*len(velocities)) 
    ax.set_xlabel(f"t [s]", fontsize = 12)
    ax.set_ylabel("v [m/s]", fontsize = 12)
    ax.legend(bbox_to_anchor=(1.2,0.5), fontsize = 12, loc="center right")
    return fig
    
def visualize_poses(cdpr:CDPR, Ts, poses):
    fig, ax = plt.subplots(constrained_layout=True, figsize = (6,2))
    positions = poses[:,:cdpr.trans_dof]
    trans_axes = cdpr.trans_axes
    title = "Bewegung des Endeffektors"
    for i in range(cdpr.trans_dof):
        ax.plot(positions[:,i],ls=_linestyles[i][1], label=f"${trans_axes[i]}$")
        
    if cdpr.rot_dof>0:
        ax1 = ax.twinx()
        orientations = poses[:,-cdpr.rot_dof:]
        rot_axes = cdpr.rot_axes
        for j in range(cdpr.rot_dof):
            ax1.plot(orientations[:,i],ls=_linestyles[j+i][1], label=f"${rot_axes[j].rotation_symbol}$")
        ax1.set_yticks(ax1.get_yticks())
        ax1.set_yticklabels(np.round(ax1.get_yticks(),3), fontsize = 12)
        ax1.set_ylabel("$\phi$ [rad]$", fontsize = 12) 
        
    ax.set_title(title, fontsize = 14)
    ax.set_xticks(ax.get_xticks())
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    xlabels = [np.round(i*Ts, len(str(Ts))) for i in xticks]
    ax.set_xticklabels(xlabels, fontsize = 12)
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(np.round(ax.get_yticks(),3), fontsize = 12)
    ax.grid(1)
    ax.set_xmargin(0)
    ax.set_xlim(-0.01*len(poses)) 
    ax.set_xlabel(f"t [s]", fontsize = 12)
    ax.set_ylabel("$r_p[m]$", fontsize = 12)
    ax.legend(bbox_to_anchor=(1.2,0.5), fontsize = 12, loc="center right")
    return fig

def visualize_tests_2D(cdpr:CDPR, successes, failures, timeouts):
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    axes_names = cdpr.axes_names
    x_min,x_max,y_min,y_max = cdpr.axes.ravel()
    b = cdpr.b
    b_left = b[:,b[0,:]==x_min]
    b_right = b[:,b[0,:]==x_max] 
    
    if failures.shape[0] !=0 and timeouts.shape[0] !=0:
        A = failures
        B = timeouts
        nrows, ncols = A.shape
        dtype={'names':['f{}'.format(i) for i in range(ncols)],
            'formats':ncols * [A.dtype]}
        C,idx1,idx2 = np.intersect1d(A.view(dtype), B.view(dtype),return_indices=True)
        failures = np.delete(A,idx1,0)
        
    title = "Bergung bei verschiedenen Startpositionen"
    bl = ax.scatter(b_left[0,:],b_left[1,:], s=60, c = "k", marker=9)
    br = ax.scatter(b_right[0,:],b_right[1,:], s=60, c = "k", marker=8)
    if successes.size != 0:
        sc = ax.scatter(successes[:,0],successes[:,1], s=60, c = "g", marker="o", label="Erfolg")
    if failures.size != 0:
        fail = ax.scatter(failures[:,0],failures[:,1], s=70, c = "r", marker="x", label="Kollision")
    if timeouts.size != 0:
        to = ax.scatter(timeouts[:,0],timeouts[:,1], s=40, c = "y", marker="D",label="nicht gestoppt")
    ax.legend( fontsize = 14, loc="upper center", ncol= 3)
    ax.set_xticks(np.arange(x_min, x_max+1, 1.0))
    ax.set_yticks(np.arange(y_min, y_max+1, 1.0))
    ax.set_xmargin(0)
    ax.set_xlabel(axes_names[0].capitalize()+" [m]", fontsize = 14)
    ax.set_ylabel(axes_names[1].capitalize()+" [m]", fontsize = 14)
    ax.set_xticklabels(ax.get_xticks(), fontsize = 12)
    ax.set_yticklabels(ax.get_yticks(), fontsize = 12)
    fig.subplots_adjust(bottom=0.15)
    return fig

def combine_figures(figs):
    num_figs = len(figs)
    fig, axs = plt.subplots(num_figs, 1, sharex= True, figsize=(6, 2*num_figs))
    if num_figs == 1:
        axs = [axs]
    for i, fig_i in enumerate(figs):
        for j, ax_i in enumerate(fig_i.axes):
            for line in ax_i.lines:
                new_line, = axs[i].plot(line.get_xdata(), line.get_ydata(), label=line.get_label())
                new_line.set_linestyle(line.get_linestyle())
                new_line.set_linewidth(line.get_linewidth())
                new_line.set_color(line.get_color())
                new_line.set_marker(line.get_marker())
                new_line.set_markersize(line.get_markersize())
            axs[i].set_xscale(ax_i.get_xscale())
            axs[i].set_yscale(ax_i.get_yscale())
            axs[i].set_xticks(ax_i.get_xticks()[1:])
            axs[i].set_xticklabels(ax_i.get_xticklabels()[1:])
            axs[i].set_ylabel(ax_i.get_ylabel(), fontsize=10)
            axs[i].set_title(ax_i.get_title(), fontsize=12)
            axs[i].set_xmargin(0)
            axs[i].grid(1)
            axs[i].legend(bbox_to_anchor=(1.25,0.5), fontsize = 12, loc="center right")
            axs[i].set_xlim(ax_i.get_xlim())
            axs[i].set_ylim(ax_i.get_ylim())
    axs[i].set_xlabel(ax_i.get_xlabel(), fontsize=10)
    fig.tight_layout()
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