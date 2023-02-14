import numpy as np
import pandas as pd
import os
def test_trajectory(cdpr,model,pose, max_timesteps):
    obs = cdpr.reset(pose)
    for i in range(len(max_timesteps)):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = cdpr.step(action)
        if done:
            break
    
    success = info["is_success"]
    end = cdpr.pose
    duration = i*cdpr.Ts
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
    for i in range(len(max_timesteps)):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = cdpr.step(action)
        poses[i,:] = cdpr.pose.ravel()
        velocities[i,:] = cdpr.pose_dot.ravel()
        accelerations[i,:] = cdpr.pose_dot_dot.ravel()
        forces[i,:] = cdpr.f.ravel()
        if done:
            break
    trajectory = {"poses": poses[:i+1,:],
                  "velocities": velocities[:i+1,:],
                  "accelerations": accelerations[:i+1,:],
                  "forces": forces[:i+1,:],
                  "success": info["is_success"],
                  "duration":i*Ts,
                  "done": done}
    return trajectory

def test_coordinates(cdpr, model, coordinates, save_freq=1, max_timesteps=10000):
    trajectories = []
    successes = []
    failures = []
    durations = []
    timeouts = []
    start_end= []
    for k, pose in enumerate(coordinates):
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
        else:
            failures.append(pose)
            if not done:
                timeouts.append(pose)
        start_end.append({"start":start,"end":end})
        durations.append(duration)
        test_data = {trajectories, 
                     successes, 
                     failures, 
                     durations, 
                     timeouts, 
                     start_end}
    return test_data

def evaluate_trajectory(poses, velocities, accelerations, forces):
    jerk = np.diff(accelerations, axis=0)
    rms_jerk = np.sqrt(np.mean(np.square(jerk)))
    max_velocity = np.max(velocities)
    rms_velocity = np.sqrt(np.mean(np.square(velocities)))
    trajectory_length = np.sum(np.sqrt(np.sum(np.diff(poses, axis=0)**2, axis=1)))
    trajectory_distance = poses[-1]-poses[0]
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