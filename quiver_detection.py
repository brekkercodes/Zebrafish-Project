import cv2
import numpy as np
from scipy.spatial.distance import euclidean

# Load video
cap = cv2.VideoCapture('input_videos/Quiver.mp4')

# Background subtractor
backSub = cv2.createBackgroundSubtractorMOG2(history=1500, varThreshold=50, detectShadows=False)

# Quivering thresholds
distance_threshold = 5
angle_threshold = 1  # Degrees threshold for detecting near-parallel alignment
min_safe_distance = 5  # Minimum distance threshold to prevent calculation issues
quiver_frame_count = 0  # Counter for continuous quiver behavior detection
quiver_persistence_threshold = 10  # Frames that confirm quiver behavior

# Time threshold (in seconds) for quiver detection
quiver_time_threshold = 3  # 3 seconds as an example for quivering to be confirmed

# Variables for time tracking
prev_time = 0  # Store the previous frame time
quiver_start_time = 0  # Store the time when quivering starts

# Function to calculate angle between two centroids
def calculate_angle_between_centroids(centroid1, centroid2):
    delta_x = centroid2[0] - centroid1[0]
    delta_y = centroid2[1] - centroid1[1]
    angle_rad = np.arctan2(delta_y, delta_x)
    angle_deg = np.degrees(angle_rad)
    # Convert angle to range [0, 180]
    angle_deg = abs(angle_deg) % 180
    return angle_deg

# Main processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get current frame time
    current_time = cv2.getTickCount() / cv2.getTickFrequency()

    # Process the current frame (convert to gray, apply blur and background subtraction)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    fg_mask = backSub.apply(blurred_frame)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    clean_fg_mask = cv2.morphologyEx(clean_fg_mask, cv2.MORPH_OPEN, kernel)

    # Detect contours and calculate centroids
    contours, _ = cv2.findContours(clean_fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_centroids = []

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                current_centroids.append((cx, cy))
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)  # Draw centroid in yellow

    # Quivering logic based on centroid count
    if len(current_centroids) == 2:
        centroid1, centroid2 = current_centroids[:2]

        # Calculate distance and angle between centroids
        distance = euclidean(centroid1, centroid2)
        angle = calculate_angle_between_centroids(centroid1, centroid2)

        # Reset the quivering frame count if the fish are apart
        if distance >= distance_threshold or angle >= angle_threshold:
            quiver_frame_count = 0

        # Check if the distance and angle meet quivering criteria
        if distance < distance_threshold and angle < angle_threshold:
            if quiver_frame_count == 0:  # If quiver is detected, set the start time
                quiver_start_time = current_time  # Record when the quiver starts
            quiver_frame_count += 1
            # Check if the quiver persists for the defined time threshold
            if current_time - quiver_start_time >= quiver_time_threshold:
                cv2.putText(frame, "Quivering Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No Quivering", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    elif len(current_centroids) == 1:
        # If only one centroid is detected, assume fish are close and quivering
        quiver_frame_count += 1
        if quiver_frame_count >= quiver_persistence_threshold:
            cv2.putText(frame, "Quivering Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    else:
        # Reset the quiver counter if no centroids or more than two centroids are found
        quiver_frame_count = 0
        cv2.putText(frame, "No Quivering", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the result with real-time information
    table_x, table_y = frame.shape[1] - 200, 20
    cv2.rectangle(frame, (table_x - 10, table_y - 10), (table_x + 180, table_y + 70), (255, 255, 255), -1)
    cv2.putText(frame, f"Centroids: {len(current_centroids)}", (table_x, table_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    if len(current_centroids) == 2:
        cv2.putText(frame, "Distance: {:.2f}".format(distance), (table_x, table_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    else:
        cv2.putText(frame, "Distance: N/A", (table_x, table_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow('Fish Detection - Quivering Detection', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
