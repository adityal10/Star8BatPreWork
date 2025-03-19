from ultralytics import YOLO
import cv2
from ultralytics.engine.results import Keypoints
from utils import get_bbox_mid_point
import numpy as np

class PlayerPose:
    def __init__(self, model_path):
        self.model = YOLO(model=model_path)

    def detect_frames(self, video_frames):
        """
        Detects the batsman on strike filtering out the other players on the field.

        Args:
            list of video frames
        Returns:
            list of player pose detections with their track id, masks and bounding box.
        """
        
        player_pose_detections = []
        
        for frame in video_frames:
            results = self.model.track(frame)[0]

            player_dict = {}

            # get frame dimensions
            height, width = frame.shape[0], frame.shape[1]
            center_y = height // 2  # Midpoint of the frame

            # ------------------------------- FOR POSE DETECTION MODEL --------------------------
            # ------------------------------- FOR POSE DETECTION MODEL --------------------------
            
            # # process each detected player
            # for box, Keypoints in zip(results.boxes, results.keypoints):
            #     print(Keypoints)
            #     break
            #     track_id = int(box.id[0]) # get player ID
            #     bounding_box = box.xyxy.tolist()[0] # get boudning box
            #     keypoint = Keypoints.xy[0] # get keypoints

            #     # calculate mid point of the bounding box
            #     mid_x, mid_y = get_bbox_mid_point(bbox=bounding_box)

            #     # condition number 2
            #     wid1, wid2 = width//2-100, width//2+100

            #     # check if player is in the upper half
            #     if (mid_y < center_y) and (wid1 < mid_x < wid2):
            #         # print(f"Player {track_id} is in the upper half. Midpoint: ({mid_x}, {mid_y}) and inbetween ({wid1} and {wid2})")
            #         player_dict[track_id] = {'keypoints': keypoint, 'bbox': bounding_box}

            # ------------------------------- FOR POSE DETECTION MODEL --------------------------
            # ------------------------------- FOR POSE DETECTION MODEL --------------------------


            for i, (box, mask) in enumerate(zip(results.boxes, results.masks)): 
                bounding_box = box.xyxy.tolist()[0] # get bounding box
                masks = mask.xy[0].tolist() # get mask points

                # calculate mid point of the bounding box
                mid_x, mid_y = get_bbox_mid_point(bbox=bounding_box)

                # condition number 2 - mid point in the range of width1 and width2
                wid1, wid2 = width//2-100, width//2+100

                # check if player is in the upper half
                if (mid_y < center_y) and (wid1 < mid_x < wid2):
                    # print(f"Player {i} is in the upper half. Midpoint: ({mid_x}, {mid_y}) and inbetween ({wid1} and {wid2})")
                    player_dict[i] = {'masks': masks, 'bbox': bounding_box}
    
            player_pose_detections.append(player_dict) # appending player detections of each frame

        return player_pose_detections


    def draw_bboxes(self, video_frames, player_pose_detections):
        """
        Draws the bounding boxes on the object of each frame.

        Args:
            List of video frames
            Players pose detections generated
        Returns:
            List of video frames with bounding box on each frame.
        """
        output_video_frames = []

        for frame, player_dict in zip(video_frames, player_pose_detections):
            # draw bounding box
            for track_id, data in player_dict.items():
                x1, y1, x2, y2 = data['bbox'] # player bounding box coordinates
                # keypoints = data['keypoints'] # player keypoints
                
                
                cv2.putText(frame, f"Player ID: {track_id}", (int(data['bbox'][0]), int(data['bbox'][1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2) # draw player bounding box

                # # Draw keypoints
                # for kp in keypoints:
                #     x, y = int(kp[0]), int(kp[1])  # Convert to integers
                #     frame = cv2.circle(frame, (x, y), keypoint_radius, keypoint_color, -1)  # -1 fills the circle
                
            output_video_frames.append(frame)

        return output_video_frames


    def mask_video_frames_binary(self, video_frames, player_pose_detections, background="black"):
        """
        Masks the object on each frame with black background and only showing the object with masks

        Args:
            List of video frames
            List of player pose detections generated
            Background (default) - `black`

        Returns:
            List of video frames with masks on each frame
        """

        output_video_frames = []

        for frame, player_dict in zip(video_frames, player_pose_detections):
            if player_dict:
                # Create a black or transparent background
                if background == "black":
                    overlay_frame = np.zeros_like(frame)  # Black background
                elif background == "transparent":
                    overlay_frame = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)  # Transparent background (RGBA)
                else:
                    raise ValueError("Invalid background option. Use 'black' or 'transparent'.")

                for track_id, data in player_dict.items():
                    masks = data['masks']

                    # Create a binary mask from the polygon points
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    polygon_points = np.array(masks, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [polygon_points], color=1)

                    if background == "black":
                        # Extract the object and place it on the black background
                        overlay_frame[mask == 1] = frame[mask == 1]
                    elif background == "transparent":
                        # Extract the object and place it on the transparent background (RGBA)
                        overlay_frame[..., :3][mask == 1] = frame[mask == 1]
                        overlay_frame[..., 3][mask == 1] = 255  # Set alpha channel to 255 (fully opaque)

                output_video_frames.append(overlay_frame)
            else:
                # If no detections, add a black or transparent frame
                if background == "black":
                    output_video_frames.append(np.zeros_like(frame))
                elif background == "transparent":
                    output_video_frames.append(np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8))

        return output_video_frames


    def mask_video_frames(self, video_frames, player_pose_detections, color=(0, 255, 0), alpha=0.5):
        """
        Masks the object on each frame

        Args:
            List of video frames
            List of player pose detections generated

        Returns:
            List of video frames with masks on each frame
        """
        output_video_frames = []
        
        for frame, player_dict in zip(video_frames, player_pose_detections):
            if player_dict:
                for track_id, data in player_dict.items():

                    overlay_frame = frame.copy()
                    masks = data['masks']
                    bbox = data['bbox']

                    # Create a binary mask from the polygon points
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    polygon_points = np.array(masks, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [polygon_points], color=1)

                    # Create a colored mask
                    colored_mask = np.zeros_like(overlay_frame)
                    colored_mask[mask == 1] = color

                    # Blend the colored mask with the original frame
                    overlay_frame = cv2.addWeighted(overlay_frame, 1, colored_mask, alpha, 0)

                    output_video_frames.append(overlay_frame)
            else:
                output_video_frames.append(frame)
                
        return output_video_frames
