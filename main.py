import cv2
import mediapipe as mp

from util_funcs.get_coordinates import get_coordinates
from util_funcs.draw_dots import draw_dots_on_image


data_source = 0 #номер камеры, или ссылка на исчтоник стрима

custom_colour = (192, 220, 90) #dark cyan (BGR)

def face_mesh_video(video_source):

    mp_drawing = mp.solutions.drawing_utils  # type: ignore
    mp_face_mesh = mp.solutions.face_mesh # type: ignore
    drawing_spec = mp_drawing.DrawingSpec(color=(custom_colour),thickness=1, circle_radius=1)
    
    org_image = None


    cap = cv2.VideoCapture(video_source)
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
        
            success, image = cap.read()
            
            
            
            if not success:
                print("Ignoring empty camera frame.")
                
                continue

            org_image = image    
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)
        
    

        # Рисуем маску на кадре
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=drawing_spec)
                    
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=drawing_spec)
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=drawing_spec)
            
        
        
            cv2.imshow('Face Mesh', image)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                return image, org_image, results
                
            
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    
    
   image, org_image, detection_results = face_mesh_video(data_source)
   
   
   coordinates_list = get_coordinates(detection_results, image)
   
   dotted_image = draw_dots_on_image(detection_results, org_image)
   
#    print(coordinates_list)
   
#    cv2.imshow('dl', dotted_image)
   
#    cv2.waitKey(0)
   

   