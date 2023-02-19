import numpy as np
import pandas as pd
import os
from tqdm.notebook import tqdm
tqdm()
def test_trajectory(cdpr,model,pose, max_timesteps):
    obs = cdpr.reset(pose)
    for t in range(len(1,max_timesteps)):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = cdpr.step(action)
        if done:
            break
    
    success = info["is_success"]
    end = cdpr.pose
    duration = t*cdpr.Ts
    return success, end, done, duration
    
def generate_trajectory(cdpr,model,pose, max_timesteps):
    n = cdpr.n
    m = cdpr.m
    Ts = cdpr.Ts
    obs = cdpr.reset(pose)
    poses = np.zeros((max_timesteps,n))
    velocities = np.zeros((max_timesteps,n))
    accelerations = np.zeros((max_timesteps,n))
    forces = np.zeros((max_timesteps,m))
    poses[0,:] = cdpr.pose.ravel()
    velocities[0,:] = cdpr.pose_dot.ravel()
    accelerations[0,:] = cdpr.pose_dot_dot.ravel()
    forces[0,:] = cdpr.f.ravel()
    for t in range(1,max_timesteps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = cdpr.step(action)
        poses[t,:] = cdpr.pose.ravel()
        velocities[t,:] = cdpr.pose_dot.ravel()
        accelerations[t,:] = cdpr.pose_dot_dot.ravel()
        forces[t,:] = cdpr.f.ravel()
        if done:
            break
    trajectory = {"poses": poses[:t+1,:],
                  "velocities": velocities[:t+1,:],
                  "accelerations": accelerations[:t+1,:],
                  "forces": forces[:t+1,:],
                  "success": info["is_success"],
                  "duration":t*Ts,
                  "done": done}
    return trajectory

def test_coordinates(cdpr, model, coordinates, save_freq=1, max_timesteps=10000):
    trajectories = []
    successes = []
    failures = []
    durations = []
    timeouts = []
    start_end= []
    for k, pose in enumerate(tqdm(coordinates)):
        start = pose
        if k%save_freq==0:
            trajectory = generate_trajectory(cdpr,model,pose, max_timesteps)
            trajectories.append(trajectory)
            success = trajectory["success"]
            done = trajectory["done"]
            end = trajectory["poses"][-1]
            duration = trajectory["duration"] 
        else:
            success, end, done, duration = test_trajectory(cdpr,model,pose, max_timesteps)
        if success:
            successes.append(pose)
        elif not done:
            timeouts.append(pose)
        else:
            failures.append(pose)
            
        start_end.append({"start":start,"end":end})
        durations.append(duration)
        test_data = {"trajectories":trajectories, 
                     "successes":successes, 
                     "failures":failures, 
                     "durations":durations, 
                     "timeouts":timeouts, 
                     "start_end":start_end}
    return test_data

def continutiy_cost(forces, f_min, f_max):
    cost = np.diff(forces)**2/(f_max-f_min)
    return cost

def evaluate_trajectory(poses, velocities, accelerations, forces):
    jerk = np.diff(accelerations, axis=0)
    rms_jerk = np.sqrt(np.mean(np.square(jerk)))
    max_velocity = np.max(velocities)
    rms_velocity = np.sqrt(np.mean(np.square(velocities)))
    trajectory_length = np.sum(np.sqrt(np.sum(np.diff(poses, axis=0)**2, axis=1)))
    trajectory_distance = np.linalg.norm(poses[-1]-poses[0])
    relative_length = trajectory_length/trajectory_distance
    rms_forces = np.sqrt(np.mean(np.square(forces)))
    max_forces = np.max(forces)
    max_velocities = np.max(velocities)
    metrics = {
        "rms_jerk": rms_jerk,
        "max_velocity": max_velocity,
        "rms_velocity": rms_velocity,
        "trajectory_length": trajectory_length,
        "trajectory_distance": trajectory_distance,
        "relative_length": relative_length,
        "rms_forces": rms_forces,
        "max_forces": max_forces,
        "max_velocities": max_velocities
    }
    return metrics

def save_results(test_data, savedir):
    df = pd.DataFrame(test_data)
    excel_path = os.path.join(savedir, "CDPR.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        df.to_excel(writer, sheet_name= "test",engine="xlsxwriter")