import cv2
import os

def save_frames_from_video(video_path, output_folder, frame_interval=1):
    """
    Extract and save frames from a video.
    
    :param video_path: Path to the input video file
    :param output_folder: Folder to save the extracted frames
    :param frame_interval: Save every 'frame_interval' frames (default is every frame)
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video was successfully opened
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second: {fps}")

    frame_index = 0
    saved_frame_count = 0

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        
        if not ret:
            break  # End of video

        # Save frame if it meets the interval condition
        if frame_index % frame_interval == 0:
            # Create a filename based on the current frame index
            frame_filename = os.path.join(output_folder, f"frame_{frame_index:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")
            saved_frame_count += 1

        frame_index += 1

    # Release the video capture object
    cap.release()
    print(f"Done. Saved {saved_frame_count} frames from the video.")

# Example usage:
video_path = 'input_videos/MVI_0052.MP4'
output_folder = 'images'
frame_interval = 30  # Save every 30th frame (approximately every second if the video is 30 FPS)

save_frames_from_video(video_path, output_folder, frame_interval)
