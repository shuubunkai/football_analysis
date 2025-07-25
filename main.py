import os
import pickle
from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import speed_and_distance_estimator
def main():
    # File paths
    video_path = 'INPUT_VIDEOS/08fd33_4.mp4'
    stub_path = 'stubs/tracks_stubs.pkl'
    output_video_path = 'output_videos/output_video.avi'

    # READ VIDEO
    video_frames = read_video(video_path)
   
    
    

    
         

    """ initate tracking """ 
    tracker = Tracker('models/best.pt')

   """ if stub|\ tracking already exists """
    if os.path.exists(stub_path):
        print("Loading tracking data from stub...")
        with open(stub_path, 'rb') as f:
            tracks = pickle.load(f)
    else:
        print("Tracking objects in video...")
        tracks = tracker.get_object_tracks(video_frames)
        # Save tracking data to stub
        os.makedirs(os.path.dirname(stub_path), exist_ok=True)
        with open(stub_path, 'wb') as f:
            pickle.dump(tracks, f)
        print("Tracking data saved.")


      #draw output
      ## Draw object tracks
    

    # get object positons 
    tracker.add_positions_to_tracks(tracks) 


    # estimate camera movement 
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,\
                                                                               read_from_stubs=True,
                                                                                 stub_path='stubs/camera_movement_stubs.pkl')  


    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    #3
    
    # Adjust player positions based on camera movement
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])


    # crop pic of a player

    """for track_id,player in tracks['players'][0].items():
        bbox = player['bbox']
        frame = video_frames[0]

        # crop the bounding box from the frame 
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # save the image 
        cv2.imwrite(f'output_videos/cropped_img.jpg', cropped_image)
        break """

    """speed and distance"""

    speed_distance_estimator = speed_and_distance_estimator()
    speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    """assign team playerrs"""

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0]
                                    ,tracks['players'][0])
    
    for frame_num , player_track in enumerate(tracks['players']):
        for player_id , track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):
      ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox')

        # Validate ball_bbox before assigning
      if ball_bbox and len(ball_bbox) == 4 and all(ball_bbox):
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
      else:
            assigned_player = None

      if assigned_player is not None and assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_id = tracks['players'][frame_num][assigned_player]['team']
           
            team_ball_control.append(team_id)
      else:
            
            team_ball_control.append(-1)  # Neutral or unknown

    team_ball_control = np.array(team_ball_control, dtype=int)


    output_video_frame = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    # draw camera movement
    output_video_frame = camera_movement_estimator.draw_camera_movement(output_video_frame, camera_movement_per_frame)

    #drAW SPEED AND DISTANCE 
    speed_distance_estimator.draw_speed_and_distance(output_video_frame, tracks)
    # SAVE VIDEO
    save_video(output_video_frame, output_video_path)
    print("Output video saved.")

if __name__ == '__main__':
    main()





