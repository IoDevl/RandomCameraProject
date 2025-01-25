import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat

mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

with pyvirtualcam.Camera(width=width, height=height, fps=30) as cam:
    print(f'Virtual camera created: {cam.device}')
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )
    
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
        model_selection=1
    )

    face_censored = False

    def blur_background(image, segmentation_mask):
        blurred = cv2.GaussianBlur(image, (55, 55), 0)
        condition = np.stack((segmentation_mask,) * 3, axis=-1) > 0.1
        output_image = np.where(condition, image, blurred)
        return output_image

    def censor_face(image, face_detection_result):
        for detection in face_detection_result.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            extra_height_up = int(height * 0.5)
            extra_height_down = int(height * 0.3)
            extra_width = int(width * 0.4)
            
            new_x = max(0, x - extra_width//2)
            new_y = max(0, y - extra_height_up)
            new_width = min(width + extra_width, w - new_x)
            new_height = min(height + extra_height_up + extra_height_down, h - new_y)
            
            face_region = image[new_y:new_y+new_height, new_x:new_x+new_width]
            if face_region.size > 0:
                pixelated = cv2.resize(face_region, (8, 8), interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(pixelated, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                pixelated = cv2.GaussianBlur(pixelated, (15, 15), 10)
                image[new_y:new_y+new_height, new_x:new_x+new_width] = pixelated

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to capture frame from camera")
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        segmentation_result = selfie_segmentation.process(image_rgb)
        hand_results = hands.process(image_rgb)
        face_results = face_detection.process(image_rgb)
        image = blur_background(image, segmentation_result.segmentation_mask)
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
        
        if face_censored and face_results.detections:
            censor_face(image, face_results)
        
        cv2.imshow(':P', image)
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cam.send(frame_rgb)
        cam.sleep_until_next_frame()
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            face_censored = not face_censored

    cap.release()
    cv2.destroyAllWindows() 