from networkx import center
from ultralytics import YOLO
import supervision as sv
import os
import pickle
import cv2
import sys
import numpy as np
import pandas as pd
sys.path.append('../') 
from utils.bbox_utils import get_center_of_bbox, get_bbox_width
from utils.bbox_utils import  get_foot_position




class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_positions_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position  

    
    
    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[])for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        #interpolate missing vals

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        ball_positions = [{1: {"bbox" :x}}for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions



    def detect_frame(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections.extend(detections_batch)  # Use extend instead of += for clarity
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks   
           
        detections = self.detect_frame(frames)

        tracks = {
            "players": [{} for _ in range(len(frames))],  # Pre-allocate dictionaries for each frame
            "referees": [{} for _ in range(len(frames))],
            "ball": [{} for _ in range(len(frames))]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert detection to supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper to player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # Process tracked objects (players and referees)
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()  # Corrected variable name from 'boux' to 'bbox'
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # Process ball detections (non-tracked objects)
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f) 

        return tracks
    
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)


        cv2.ellipse(
     frame,
     (x_center, y2),                           # center
     (int(width), int(0.35 * width)),          # axes
     0,                                        # angle of rotation 
     -45,                                       # startAngle
     235,                                      # endAngle
     color,                                    # color
     3,                                        # thickness
     cv2.LINE_4                                # lineType
        )  
        rectangle_width = 40 
        rectangle_height = 20 
        x1_rect  =  x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2)+15
        y2_rect = (y2 + rectangle_height//2)+15

        if track_id is not None:


            cv2.rectangle(frame,
                        (int(x1_rect),int(y1_rect)),
                        (int(x2_rect),int(y2_rect)),
                        color,
                        cv2.FILLED) 

            x1_text  = x1_rect + 11
            if track_id > 99:
              x1_text -=10

            cv2.putText(
            frame,
            f"{track_id}",
            (int(x1_text),int(y1_rect+15)),
            cv2.FONT_HERSHEY_COMPLEX,
            0.7,
            (0,0,0),
            2
            )



        return frame  



    def draw_triangle(self,frame,bbox,color):
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])
        cv2.drawContours(frame, [triangle_points], 0, color,-1)# -1 is filled
        cv2.drawContours(frame, [triangle_points], 0, (0,0,0), 2 ) # draw border


        return frame
    

    


    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        #draw seemi_transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay,(1350, 850),(1900,970),(255,255,255),cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # get the number of times each team has the ball
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]


        team_1= team_1_num_frames/(team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames + team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400, 900),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400, 950),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 0), 3)

        return frame




    
  
  
   





    def draw_annotations(self, frames, tracks,team_ball_control):
        output_video_frames = []

        for frame_num, frame in enumerate(frames):
           frame = frame.copy()

           player_dict = tracks["players"][frame_num]
           ball_dict = tracks["ball"][frame_num]
           referee_dict = tracks["referees"][frame_num] 

         # draw players

           for track_id, player in player_dict.items():
               color = player.get("team_color",(0,0,255))
               
               frame = self.draw_ellipse(frame,player["bbox"],color,track_id)

               if player.get('has_ball',False):
                 frame = self.draw_triangle(frame,player["bbox"],(0,0,255))

            #draw referees
           for _, referee in referee_dict.items():
               frame = self.draw_ellipse(frame,referee["bbox"],(0,255,255),)  

             
            #draw ball
           for track_id, ball in ball_dict.items():
               frame = self.draw_triangle(frame,ball["bbox"],(0,255,0))
               
           # draw team ball control
           frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)


           output_video_frames.append(frame)
      
        return output_video_frames


        