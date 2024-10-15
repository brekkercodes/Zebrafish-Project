import cv2

def process_video_for_edges_and_segmentation(video_path, frame_interval=30):
    """
    Apply filters to video, detect fish edges, and perform segmentation.
    This function processes the video and displays the results without saving images.
    
    :param video_path: Path to the input video file
    :param frame_interval: Process every 'frame_interval' frames (default is every 30th frame)
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was successfully opened
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_index = 0

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        
        if not ret:
            break  # End of video

        # Process every nth frame according to the frame_interval
        if frame_index % frame_interval == 0:
            # Resize the frame (optional for faster processing)
            frame_resized = cv2.resize(frame, (640, 360))

            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian Blur to reduce noise
            blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

            # Apply Canny Edge Detection to detect edges
            edges = cv2.Canny(blurred_frame, threshold1=10, threshold2=80)

            # Perform Thresholding for segmentation
            _, thresh = cv2.threshold(blurred_frame, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Find contours (fish edges) from the segmented image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw the contours on the original frame
            contour_frame = frame_resized.copy()
            cv2.drawContours(contour_frame, contours, -1, (0, 255, 0), 2)

            # Display the original frame, edge-detected frame, and contour frame side by side
            combined_frame = cv2.hconcat([frame_resized, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), contour_frame])
            cv2.imshow("Fish Edge Detection and Segmentation", combined_frame)

        frame_index += 1

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing complete.")

# Example usage:
video_path = 'input_videos/MVI_0052.MP4'
frame_interval = 30  # Process every 30th frame

process_video_for_edges_and_segmentation(video_path, frame_interval)
