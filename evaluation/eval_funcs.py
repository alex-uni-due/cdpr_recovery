import numpy as np

def test_trajectory(cdpr,model,pose, max_timesteps):
    obs = cdpr.reset(pose)
    for t in range(max_timesteps):
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

def continuity_cost(a, a_min, a_max):
    cost = 100*(np.abs(np.diff(a,axis=0))/(a_max-a_min))
    cost_mean = np.mean(cost)
    return cost_mean

def find_final_threshold_index(signal, threshold):
    idx = len(signal)
    check_if_under_threshold = True
    for i, value in enumerate(signal):
        if check_if_under_threshold:
            if abs(value) <= threshold:
                check_if_under_threshold = False
                idx = i
        else:
            if abs(value) >threshold:
                idx = len(signal)
                check_if_under_threshold = True
    return idx

def find_not_rising_index(signal):
    idx = len(signal)
    is_rising = True
    for i, value in enumerate(signal):
        if i==0:
            continue
        if value==signal[i-1]:
            continue
        if is_rising:
            if (value-signal[i-1])<0:
                is_rising = False
                idx = i
        elif (value-signal[i-1])>0:
            is_rising = True
            idx = i
    return idx
            
def evaluate_translation_trajectory(positions, velocities, accelerations):
    jerk = np.diff(accelerations, axis=0)
    rms_jerk = np.sqrt(np.mean(np.square(jerk)))
    max_velocity = np.max(velocities)
    rms_velocity = np.sqrt(np.mean(np.square(velocities)))
    trajectory_length = np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1)))
    trajectory_distance = np.linalg.norm(positions[-1]-positions[0])
    relative_length = trajectory_length/trajectory_distance
    metrics = {
        "rms_jerk": rms_jerk,
        # "max_velocity": max_velocity,
        # "rms_velocity": rms_velocity,
        "trajectory_length": trajectory_length,
        "trajectory_distance": trajectory_distance,
        "relative_length": relative_length,
    }
    return metrics

def evaluate_force_trajectory(forces):
    force_gradients = np.diff(forces, axis=0)
    rms_forces = np.sqrt(np.mean(np.square(forces)))
    rms_force_gradients = np.sqrt(np.mean(np.square(force_gradients)))
    max_force_delta = np.max(force_gradients)
    # max_force = np.max(forces)
    metrics = {
        "rms_forces": rms_forces,
        "rms_force_gradients":rms_force_gradients,
        # "max_force": max_force,
        "max_force_delta": max_force_delta,
    }
    return metrics

def evaluate_rotation_trajectory(orientations, angular_velocities, angular_accelerations):
    jerk = np.diff(angular_accelerations, axis=0)
    rms_jerk = np.sqrt(np.mean(np.square(jerk)))
    max_velocity = np.max(angular_velocities)
    rms_velocity = np.sqrt(np.mean(np.square(angular_velocities)))
    total_rotation = np.sum(np.sqrt(np.sum(np.diff(orientations, axis=0)**2, axis=1)))
    metrics = {
        "rms_angular_jerk": rms_jerk,
        "max_angular_velocity": max_velocity,
        "rms_angular_velocity": rms_velocity,
        "total_rotation":total_rotation
    }
    return metrics

def average_distance_to_centroid(safe_positions):
    """
    Calculates the average distance to centroid for a set of safe positions.
    :param safe_positions: A numpy array of shape (n, d) containing n safe positions in d dimensions.
    :return: The average distance to centroid.
    """
    if len(safe_positions)==0:
        return 0.
    
    # Calculate the centroid
    centroid = np.mean(safe_positions, axis=0)

    # Calculate the distances to centroid
    distances = np.sqrt(np.sum((safe_positions - centroid)**2, axis=1))

    # Calculate the average distance to centroid
    avg_distance = np.mean(distances)

    return avg_distance