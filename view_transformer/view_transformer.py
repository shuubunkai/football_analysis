import numpy as np
import cv2

class ViewTransformer:
    def __init__(self):
        pass
        field_length = 68
        field_width = 23.32


        self.pixel_vertices = np.array([
            [110,1035],
            [265,275],
            [910,259],
            [1641,915]

        ]) 

        self.target_vertices = np.array([
           [0,field_length],
           [0,0],
           [field_length, 0],
           [field_length, field_width]

       ]) 
        
        self.pixel_vertices= self.pixel_vertices.astype(np.float32) 
        self.target_vertices = self.target_vertices.astype(np.float32)


        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    

    def transform_point(self,point):
        p = (int(point[0]),int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices,p,False) >= 0 
        if not is_inside:
            return None

        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshaped_point,self.perspective_transformer)
        return transform_point.reshape(-1,2)



    def add_transformed_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_trasnformed = self.transform_point(position)
                    if position_trasnformed is not None:
                        position_trasnformed = position_trasnformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed

        

    