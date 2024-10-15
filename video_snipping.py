import cv2
import os

def split_video_into_clips(video_path, output_folder, clip_duration=10):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the frames per second (fps) and total frame count
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    print(f"Total video duration: {video_duration / 60:.2f} minutes")
    
    # Calculate how many frames correspond to the clip duration (in seconds)
    clip_frames = int(clip_duration * fps)
    clip_index = 0
    current_frame = 0

    while current_frame < total_frames:
        # Define the output video filename
        output_filename = os.path.join(output_folder, f'clip_top_{clip_index:04d}.mp4')

        # Define the video writer with the same properties as the original video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        out = cv2.VideoWriter(output_filename, fourcc, fps, 
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        # Write frames for the current clip
        for _ in range(clip_frames):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current_frame += 1

        # Release the current clip
        out.release()
        clip_index += 1

        print(f"Saved clip {clip_index} as {output_filename}")
    
    # Release the video capture object
    cap.release()
    print("Video processing complete.")

# Example usage
video_path = 'input_videos/13B-16B Top 1.MP4'
output_folder = 'snipped_videos'
split_video_into_clips(video_path, output_folder)
