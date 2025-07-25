from mediapipe.framework.formats import landmark_pb2
import cv2


landmark_indicies = {
    'nose':[1, 2, 4, 5, 6, 19, 275, 278, 294, 168, 45, 48, 440, 64, 195, 197, 326, 327, 344, 220, 94, 97, 98, 115],
    'lips':[0, 267, 269, 270, 13, 14, 17, 402, 146, 405, 409, 415, 291, 37, 39, 40, 178, 308, 181, 310, 311, 312, 
            185, 314, 317, 318, 61, 191, 321, 324, 78, 80, 81, 82, 84, 87, 88, 91, 95, 375],
    'right eyebrow':[65, 66, 70, 105, 107, 46, 52, 53, 55, 63],
    'left eyebrow': [293, 295, 296, 300, 334, 336, 276, 282, 283, 285],
    'right eye': [160, 33, 161, 163, 133, 7, 173, 144, 145, 246, 153, 154, 155, 157, 158, 159],
    'left eye':  [384, 385, 386, 387, 388, 390, 263, 362, 398, 466, 373, 374, 249, 380, 381, 382],
    'face_oval': [132, 389, 136, 10, 397, 400, 148, 149, 150, 21, 152, 284, 288, 162, 297, 172, 176, 
                  54, 58, 323, 67, 454, 332, 338, 93, 356, 103, 361, 234, 109, 365, 379, 377, 378, 251, 127]
}

keys = ['nose', 'lips', 'right eyebrow', 'left eyebrow', 'right eye', 'left eye', 'face_oval']

def draw_dots_on_image(detection_results, image):

    for face_landmark in detection_results.multi_face_landmarks:
                lms = face_landmark.landmark
                
                coords_list = {}
                for key in keys:
                    temp = {}
                    for index in landmark_indicies[key]:
                        x = int(lms[index].x * image.shape[1])
                        y = int(lms[index].y * image.shape[0])
                        temp[index] = (x,y)
                    
                    coords_list[key] = temp    
                    
                    for index in landmark_indicies[key]:
                        cv2.circle(image, (coords_list[key][index][0], coords_list[key][index][1]), 2,(0,255,0), -1)
            
                    
            
                # cv2.imshow('t', image)
                    
                # cv2.waitKey(0)
                
                return image
            
def draw_dots_on_image_for_async(detection_results, image):

    face_landmarks_list = detection_results.face_landmarks
  

  
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])
                    
        coords_list = {}
        for key in keys:
                    temp = {}
                    for index in landmark_indicies[key]:
                        x = int(face_landmarks_proto.landmark[index].x * image.shape[1])
                        y = int(face_landmarks_proto.landmark[index].y * image.shape[0]) 
                        temp[index] = (x,y)
                    
                    coords_list[key] = temp    
                    
                    for index in landmark_indicies[key]:
                        cv2.circle(image, (coords_list[key][index][0], coords_list[key][index][1]), 2,(0,255,0), -1)
            
                    
            
                # cv2.imshow('t', image)
                    
                # cv2.waitKey(0)
                
        return image