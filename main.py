from utils import read_video, save_video, create_side_by_side_video
import cv2
from ultralytics import YOLO
from player_tracker import PlayerPose

import streamlit as st

def main():

    # video path input
    video_path = 'video_clips/pull2.mp4'
    video_frames = read_video(video_path) # output format (720, 1280, 3) = ( height, width, BGR (color) )

    model_path = 'yolo11n-seg.pt' # model path

    ## GET AND STORE PLAYER POSE DETECTIONS
    model = PlayerPose(model_path=model_path)
    player_pose_detections = model.detect_frames(video_frames=video_frames)


    ## DRAW PLAYER BOUNDING BOXES
    output_video_frames = model.draw_bboxes(video_frames=video_frames, player_pose_detections=player_pose_detections)

    ## DRAW PLAYER MASKS
    output_video_frames = model.mask_video_frames_binary(video_frames=output_video_frames, player_pose_detections=player_pose_detections)

    # display_side_by_side_pillow(video_frames, output_video_frames)
    
    # Create a side-by-side video
    output_video_path = "output_videos/output_side_by_side.mp4"
    create_side_by_side_video(video_frames, output_video_frames, output_video_path, fps=10)
    # save_video(output_video_frames, 'output_videos/output_video.avi')


if __name__=="__main__":
    main()
