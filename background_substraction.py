import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture('snipped_videos/clip_top_0005.mp4')

# Create background subtractor (MOG2)
backSub = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=25, detectShadows=False)

# Structuring element for noise removal (morphological operations)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

prev_gray = None  # Store previous frame for optical flow

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Convert to grayscale and blur the frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Step 2: Apply background subtraction
    fg_mask = backSub.apply(blurred_frame)

    # Step 3: Apply morphological operations to clean the mask
    clean_fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    clean_fg_mask = cv2.morphologyEx(clean_fg_mask, cv2.MORPH_OPEN, kernel)

    # Step 4: Find contours of the fish
    contours, _ = cv2.findContours(clean_fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Loop over each contour and draw bounding boxes
    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter based on contour area
        MIN_CONTOUR_AREA = 500  # Minimum size of a contour (in pixels)
        MAX_CONTOUR_AREA = 5000  # Maximum size of a contour (optional)
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue  # Skip small/large contours (likely noise)

        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate aspect ratio
        aspect_ratio = float(w) / h
        # Aspect ratio filtering (fish-like shapes)
        if aspect_ratio < 0.5 or aspect_ratio > 3:
            continue  # Skip non-fish objects

        # Draw bounding box around detected fish
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Fish", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Step 6: Calculate optical flow if the previous frame is available
    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Visualize optical flow (for debugging)
        hsv = np.zeros_like(frame)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Display optical flow
        cv2.imshow('Optical Flow', flow_frame)

    # Update previous gray frame for optical flow calculation
    prev_gray = gray_frame.copy()

    # Step 7: Display the result (with bounding boxes)
    cv2.imshow('Fish Detection with Bounding Boxes', frame)
    cv2.imshow('Foreground Mask', clean_fg_mask)

    # Exit loop on 'q' key press
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
