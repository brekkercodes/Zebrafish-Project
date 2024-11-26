import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import trajectorytools as tt
import trajectorytools.animation as ttanimation
import trajectorytools.socialcontext as ttsocial
from trajectorytools.constants import test_raw_trajectories_path

# Load test trajectories as a numpy array of locations
data = np.load('input_videos/session_clip_new_0002_2/trajectories/without_gaps.npy', allow_pickle=True).item()
trajectories = data['trajectories']

# Check the shape of the loaded trajectories
print("Shape of trajectories:", trajectories.shape)  # Should be (19500, 2, 2)

# Step 1: Verify the data type is compatible
trajectories = trajectories.astype(float)  # Ensure all values are float type

# Step 2: Try initializing Trajectories with error handling
try:
    traj = tt.Trajectories.from_positions(trajectories, smooth_params={"sigma": 1})
    print("Trajectories initialized successfully.")
except Exception as e:
    print("Error during Trajectories initialization:", e)

# Step 3: Synthetic Data Test
# Generate synthetic data in the same shape as `trajectories` to test compatibility
test_trajectories = np.random.rand(19500, 2, 2).astype(float)

try:
    traj_test = tt.Trajectories.from_positions(test_trajectories, smooth_params={"sigma": 1})
    print("Synthetic data initialized successfully.")
except Exception as e:
    print("Error with synthetic data:", e)

# Step 4: Convert `trajectories` to a list of arrays (alternative method)
# Sometimes a list of 2D arrays (one per frame) may work better
trajectories_list = [frame for frame in trajectories]

try:
    traj_list_format = tt.Trajectories.from_positions(np.array(trajectories_list), smooth_params={"sigma": 1})
    print("Trajectories initialized successfully with list format.")
except Exception as e:
    print("Error with list format initialization:", e)

# If Trajectories initialized successfully, proceed with further processing
if 'traj' in locals():
    # Assuming a circular arena, populate center and radius keys
    center, radius = traj.estimate_center_and_radius_from_locations()
    traj.origin_to(center)  # Center trajectories around estimated center
    traj.new_length_unit(radius)  # Normalize location by radius
    traj.new_time_unit(32, "second")  # Change time units to seconds

    # Find smoothed trajectories, velocities, and accelerations
    in_border = []
    for positions_in_frame in traj.s:
        if positions_in_frame.shape[0] >= 4:
            in_border_frame = ttsocial.in_alpha_border(positions_in_frame, alpha=5)
        else:
            in_border_frame = np.zeros(positions_in_frame.shape[0], dtype=bool)
        in_border.append(in_border_frame)

    in_border = np.array(in_border)  # Convert to numpy array

    # Animation setup for fish on the border
    colornorm = Normalize(vmin=0, vmax=3, clip=True)
    mapper = ScalarMappable(norm=colornorm, cmap="RdBu")
    color = mapper.to_rgba(in_border)

    anim1 = ttanimation.scatter_vectors(traj.s, velocities=traj.v, k=0.3)
    anim2 = ttanimation.scatter_ellipses_color(traj.s, traj.v, color)
    anim = anim1 + anim2

    anim.prepare()
    anim.show()

# Parameters
distance_threshold = 300.0  # Max distance for chasing (in pixels)
direction_threshold = 0.80  # Minimum cosine similarity to indicate chasing
min_chasing_duration = 0.5  # Minimum duration for counting chasing (in seconds)

# Frame duration in seconds (assuming 65 fps)
frame_duration = 1 / 65

# Initialize variables to store chasing data
chasing_events = []
current_event = None  # Placeholder for ongoing chasing events

# Iterate over each frame to detect chasing
for frame in range(len(traj.s)):
    pos_A, pos_B = traj.s[frame][0], traj.s[frame][1]
    vel_A = traj.v[frame][0]

    # Calculate distance and direction
    distance_AB = np.linalg.norm(pos_B - pos_A)
    direction_AB = (pos_B - pos_A) / distance_AB if distance_AB != 0 else np.array([0, 0])

    # Calculate cosine similarity between velocity and direction
    cosine_similarity = np.dot(vel_A, direction_AB) / np.linalg.norm(vel_A) if np.linalg.norm(vel_A) != 0 else 0

    # Determine if chasing condition is met
    is_chasing = (distance_AB < distance_threshold) and (cosine_similarity > direction_threshold)

    # Check if chasing started or ended
    if is_chasing:
        if current_event is None:
            # Start of a new chasing event
            current_event = {"start_frame": frame}
    else:
        if current_event is not None:
            # End of the ongoing chasing event
            current_event["end_frame"] = frame - 1
            # Only add the event if its duration is at least 1.5 seconds
            event_duration = (current_event["end_frame"] - current_event["start_frame"] + 1) * frame_duration
            if event_duration >= min_chasing_duration:
                chasing_events.append(current_event)
            current_event = None

# If a chasing event is ongoing at the end of the loop, finalize it
if current_event is not None:
    current_event["end_frame"] = len(traj.s) - 1
    # Check duration before adding
    event_duration = (current_event["end_frame"] - current_event["start_frame"] + 1) * frame_duration
    if event_duration >= min_chasing_duration:
        chasing_events.append(current_event)

# Calculate timestamps and durations for each chasing event
chasing_info = []
for event in chasing_events:
    start_time = event["start_frame"] * frame_duration
    end_time = (event["end_frame"] + 1) * frame_duration  # Add 1 to include the end frame
    duration = end_time - start_time
    chasing_info.append({
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration
    })

# Output the results
num_chasing_events = len(chasing_info)
total_chasing_duration = sum(event["duration"] for event in chasing_info)

print("Number of chasing events:", num_chasing_events)
print("Total chasing duration (seconds):", total_chasing_duration)
print("Chasing events details (timestamps in seconds):")
for i, event in enumerate(chasing_info, start=1):
    print(f"Event {i}: Start={event['start_time']:.2f}s, End={event['end_time']:.2f}s, Duration={event['duration']:.2f}s")
# Create a DataFrame from chasing_info
chasing_df = pd.DataFrame(chasing_info)

# Save the DataFrame to an Excel file
output_file = "chasing_behavior.xlsx"
chasing_df.to_excel(output_file, index=False)


