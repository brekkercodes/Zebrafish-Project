import cv2
import numpy as np
from skimage.morphology import skeletonize

# Load video
cap = cv2.VideoCapture('input_videos/Quiver.mp4')

# Create background subtractor with optimized parameters
backSub = cv2.createBackgroundSubtractorMOG2(history=1500, varThreshold=50, detectShadows=False)

# Function to apply skeletonization to the cleaned foreground mask
def extract_skeleton(fg_mask):
    binary_mask = fg_mask // 255
    skeleton = skeletonize(binary_mask).astype(np.uint8) * 255
    return skeleton

# Function to detect the endpoints of the skeleton (head and tail)
def get_feature_points(skeleton):
    endpoints = []
    for y in range(1, skeleton.shape[0] - 1):
        for x in range(1, skeleton.shape[1] - 1):
            if skeleton[y, x] == 255:
                neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2]) - 255
                if neighbors == 255:  # endpoint
                    endpoints.append((x, y))
    return endpoints

# Main video processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Background Subtraction and Preprocessing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    fg_mask = backSub.apply(blurred_frame)

    # Step 2: Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    clean_fg_mask = cv2.morphologyEx(clean_fg_mask, cv2.MORPH_OPEN, kernel)

    # Step 3: Skeleton Extraction
    skeleton = extract_skeleton(clean_fg_mask)

    # Step 4: Detect Contours and Fish Centroids
    contours, _ = cv2.findContours(clean_fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_centroids = []

    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small contours
            # Calculate the centroid of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                current_centroids.append((cx, cy))

            # Step 5: Skeleton and Feature Points (Head/Tail)
            feature_points = get_feature_points(skeleton)

            # Identify head by choosing point closest to centroid
            if feature_points:
                head = min(feature_points, key=lambda p: np.linalg.norm(np.array(p) - np.array((cx, cy))))
                cv2.circle(frame, head, 5, (0, 0, 255), -1)  # Draw head in red

    # Step 6: Draw Centroids and Heads
    for centroid in current_centroids:
        cv2.circle(frame, centroid, 5, (0, 255, 255), -1)  # Draw centroid in yellow

    # Display the result
    cv2.imshow('Fish Detection - Centroid and Head', frame)
    # cv2.imshow('Foreground Mask', clean_fg_mask)
    #cv2.imshow('Skeleton and Feature Points', skeleton)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
