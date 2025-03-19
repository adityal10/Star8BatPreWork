import cv2
import numpy as np
from PIL import Image


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    return frames

def read_video_st(uploaded_video):
    # Save the uploaded video to a temporary file
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_video.read())
    
    # Open the video file
    cap = cv2.VideoCapture("temp_video.mp4")
    # cap = cv2.VideoCapture(video_path)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    return frames


def get_bbox_mid_point(bbox):
    x_min, y_min, x_max, y_max = bbox
    midpoint_x = (x_min + x_max) / 2
    midpoint_y = (y_min + y_max) / 2
    return midpoint_x, midpoint_y

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))

    for frame in output_video_frames:
        out.write(frame)
    out.release()

def create_side_by_side_video(original_frames, overlay_frames, output_path, fps=30):
    """
    Create a video with original and overlay frames side by side.

    Args:
        List of original video frames.
        List of overlay output video frames.
        Path to save the output video.
        Frames per second for the output video.
    """
    # Get the dimensions of the original frames
    height, width = original_frames[0].shape[:2]

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    combined_width = width * 2  # Double the width for side-by-side frames
    combined_height = height
    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))

    # Define text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    font_thickness = 2
    text_offset = 20  # Offset from the top-left corner

    for original_frame, overlay_frame in zip(original_frames, overlay_frames):
        # Resize overlay frame to match the original frame's dimensions (if necessary)
        overlay_frame_resized = cv2.resize(overlay_frame, (width, height))

        # Concatenate frames side by side
        combined_frame = np.hstack((original_frame, overlay_frame_resized))

        # Add text labels
        cv2.putText(combined_frame, "Original", (text_offset, text_offset), font, font_scale, font_color, font_thickness)
        cv2.putText(combined_frame, "Overlayed", (width + text_offset, text_offset), font, font_scale, font_color, font_thickness)

        # Write the combined frame to the video file
        out.write(combined_frame)

        # Display the combined frame in real-time (optional)
        cv2.imshow("Side by Side Video", combined_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    # Release the VideoWriter and close windows
    out.release()
    cv2.destroyAllWindows()