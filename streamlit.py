import streamlit as st
from utils import read_video, save_video, create_side_by_side_video
from player_tracker import PlayerPose

def main():
    st.title("Player Pose Detection and Mask Overlay")
    st.write("Upload a video to generate a side-by-side output with player pose detection and mask overlay.")

    # File uploader for video input
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save the uploaded video to a temporary file
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Read the video frames
        video_frames = read_video("temp_video.mp4")

        # Load the model
        model_path = 'yolo11n-seg.pt'
        model = PlayerPose(model_path=model_path)

        # Get player pose detections
        with st.spinner("Detecting player poses..."):
            player_pose_detections = model.detect_frames(video_frames=video_frames)

        # Draw player bounding boxes
        with st.spinner("Drawing bounding boxes..."):
            output_video_frames = model.draw_bboxes(video_frames=video_frames, player_pose_detections=player_pose_detections)

        # Draw player masks
        with st.spinner("Drawing player masks..."):
            output_video_frames = model.mask_video_frames_binary(video_frames=output_video_frames, player_pose_detections=player_pose_detections)

        # Create a side-by-side video
        output_video_path = "output_videos/output_side_by_side.mp4"
        with st.spinner("Generating side-by-side video..."):
            create_side_by_side_video(video_frames, output_video_frames, output_video_path, fps=30)

        # Display the side-by-side video
        st.write("### Side-by-Side Output Video")
        video_file = open("output_videos/output_side_by_side.mp4", "rb")
        video_bytes = video_file.read()
        st.video(video_bytes)

        # Provide a download link for the output video
        with open(output_video_path, "rb") as f:
            video_bytes = f.read()
        st.download_button(
            label="Download Output Video",
            data=video_bytes,
            file_name="output_side_by_side.mp4",
            mime="video/mp4"
        )

if __name__ == "__main__":
    main()