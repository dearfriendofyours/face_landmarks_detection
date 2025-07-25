import cv2
import mediapipe as mp
import time

from util_funcs.draw_landmarks import draw_landmarks_on_image
from util_funcs.get_coordinates import get_coordinates_for_async
from util_funcs.draw_dots import draw_dots_on_image_for_async

#Путь к модели. Для корректной работы нужен полный путь
model_path ='D:\\computer_vision\\face_analyser\\models\\face_landmarker.task' 


    
class Landmarker():
    def __init__(self):
        self.result = None
        self.landmarker = mp.tasks.vision.FaceLandmarker
        self.createLandmarker()
        
    def createLandmarker(self):
        
        def update_result(result, output_image, timestamp_ms):
            self.result = result
            
        options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        num_faces = 1,
        min_face_detection_confidence = 0.5,
        min_face_presence_confidence = 0.5,
        result_callback=update_result)
        
        self.landmarker = self.landmarker.create_from_options(options)
        
    def detect_async(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        self.landmarker.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))
        
    def close(self):
        self.landmarker.close()     
        
  

      
def main():
    
    cap = cv2.VideoCapture(0)
    
    face_landmarker = Landmarker()
    
   


    while True:
        
        
        ret, frame = cap.read()
        
        if not ret:
            print('There is no video data yet')
            continue
        
        frame = cv2.flip(frame, 1)
        
        face_landmarker.detect_async(frame)
        
        if face_landmarker.result is not None:
            annotated_image = draw_landmarks_on_image(frame, face_landmarker.result)
            cv2.imshow('Frame', annotated_image)
        
            if cv2.waitKey(1) == ord('q'):
                cap. release()

                cv2.destroyAllWindows() 
                
                face_landmarker.close()
                
                return frame, annotated_image, face_landmarker.result
            
                
        else:
            print("There are no detections yet, showing original video")
        
            cv2.imshow('Frame', frame)
            
            if cv2.waitKey(1) == ord('q'):
                
                cap. release()

                cv2.destroyAllWindows() 
                
                face_landmarker.close()
                
                return frame, None, None
        
        
        
   

if __name__ == "__main__":
   
   
   
   original_image, annotated_image, detection_results = main()
   
   coordinates_list = get_coordinates_for_async(detection_results, annotated_image)
   
   dotted_image = draw_dots_on_image_for_async(detection_results, original_image)
   
#    cv2.imshow("dotted image", dotted_image)
   
#    cv2.waitKey(0)
   
   
   