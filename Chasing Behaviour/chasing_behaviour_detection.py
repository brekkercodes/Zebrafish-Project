import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trajectorytools as tt  # Ensure this is installed and compatible

# Step 1: Load the .npy file and extract trajectories
data = np.load('input_videos/session_clip_new_0002_2/trajectories/without_gaps.npy', allow_pickle=True).item()
trajectories = data['trajectories']
# Step 2: Check shape and get the number of frames and fish
frame_count, fish_count, _ = trajectories.shape

# Step 3: Convert trajectories into x and y coordinates DataFrames
x_coords = pd.DataFrame(trajectories[:, :, 0], columns=[f'fish_{i+1}' for i in range(fish_count)])
y_coords = pd.DataFrame(trajectories[:, :, 1], columns=[f'fish_{i+1}' for i in range(fish_count)])


# Step 4: Plot the trajectories for visualization (optional)
plt.figure(figsize=(10, 10))
for i in range(fish_count):
    plt.plot(x_coords[f'fish_{i+1}'], y_coords[f'fish_{i+1}'], label=f'Fish {i+1}')
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.title("Zebrafish Trajectories")
plt.show()
 
# Synthetic data example with the same shape as your trajectories
test_trajectories = np.random.rand(19500, 2, 2).astype(float)

try:
    traj = tt.Trajectories.from_positions(test_trajectories, smooth_params={"sigma": 1})
    print("Test trajectories initialized successfully.")
except Exception as e:
    print("Error with synthetic data:", e)

distances = traj.distance_to()
print("Distances:\n", distances)

# Step 5: Initialize the Trajectories object with trajectories data
# Initialize with the actual trajectories data and parameters
params = {
    'frames_per_second': data['frames_per_second'],  # Example: 65 FPS
    'body_length': data['body_length'],              # Example: 293.0
}

try:
    # Initialize Trajectories with the trajectories data and params
    traj = tt.Trajectories(trajectories, params=params)

    # Step 6: Example Analysis
    distances = traj.distance()
    print("Distances:\n", distances)

    speeds = traj.speed()
    print("Speeds:\n", speeds)

    # Proximity detection (optional)
    proximity_events = traj.proximity(threshold=5)
    print("Proximity Events:\n", proximity_events)

except Exception as e:
    print("Error during Trajectories initialization or analysis:", e)
