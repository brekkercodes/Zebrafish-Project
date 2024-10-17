import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture('snipped_videos/clip_top_0050.mp4')

# Create background subtractor
backSub = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=30, detectShadows=False)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Preprocess the frame (convert to grayscale and apply background subtraction)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    fg_mask = backSub.apply(blurred_frame)

    # Step 2: Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    clean_fg_mask = cv2.morphologyEx(clean_fg_mask, cv2.MORPH_OPEN, kernel)

    # Step 3: Find contours
    contours, _ = cv2.findContours(clean_fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: Process each detected contour
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small contours
            # Draw the contour
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

            # Step 5: Compute the extreme points
            leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
            rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
            topmost = tuple(contour[contour[:, :, 1].argmin()][0])
            bottommost = tuple(contour[contour[:, :, 1].argmax()][0])

            # Step 6: Calculate the centroid of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            # Use the centroid to differentiate head and tail (extreme points farthest from centroid)
            # Euclidean distance from centroid to each extreme point
            distances = {
                "leftmost": np.linalg.norm(np.array([cx, cy]) - np.array(leftmost)),
                "rightmost": np.linalg.norm(np.array([cx, cy]) - np.array(rightmost)),
                "topmost": np.linalg.norm(np.array([cx, cy]) - np.array(topmost)),
                "bottommost": np.linalg.norm(np.array([cx, cy]) - np.array(bottommost))
            }

            # Find the two farthest points (likely head and tail)
            sorted_distances = sorted(distances.items(), key=lambda x: x[1], reverse=True)
            head = sorted_distances[0][0]
            tail = sorted_distances[1][0]

            # Assign head and tail points
            if head == "leftmost":
                head_point = leftmost
            elif head == "rightmost":
                head_point = rightmost
            elif head == "topmost":
                head_point = topmost
            else:
                head_point = bottommost

            if tail == "leftmost":
                tail_point = leftmost
            elif tail == "rightmost":
                tail_point = rightmost
            elif tail == "topmost":
                tail_point = topmost
            else:
                tail_point = bottommost

            # Step 7: Draw the head and tail on the frame
            cv2.circle(frame, head_point, 5, (0, 0, 255), -1)  # Head in red
            cv2.circle(frame, tail_point, 5, (255, 0, 0), -1)  # Tail in blue

            # Display the centroid as well (for reference)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)  # Centroid in yellow

    # Step 8: Display the result
    cv2.imshow('Fish Detection with Head and Tail', frame)
    cv2.imshow('Foreground Mask', clean_fg_mask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
