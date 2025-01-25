import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat
import math

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
        min_detection_confidence=0.8,
        min_tracking_confidence=0.6
    )

    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )
    
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
        model_selection=1
    )

    face_censored = False
    last_face_detection = None
    show_measurements = False

    def calculate_distance(p1, p2, image_shape):
        h, w = image_shape[:2]
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        
        pixel_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if pixel_distance == 0:
            return 0
        
        cm_per_pixel = 8.5 / (w * 0.2)
        return pixel_distance * cm_per_pixel

    def draw_hand_landmarks_with_measurements(image, hand_landmarks):
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )
        
        if show_measurements:
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)
            ]
            
            for start_idx, end_idx in connections:
                start_point = hand_landmarks.landmark[start_idx]
                end_point = hand_landmarks.landmark[end_idx]
                
                distance_cm = calculate_distance(start_point, end_point, image.shape)
                
                x_mid = int((start_point.x + end_point.x) * image.shape[1] / 2)
                y_mid = int((start_point.y + end_point.y) * image.shape[0] / 2)
                
                cv2.putText(
                    image,
                    f'{distance_cm:.1f}cm',
                    (x_mid, y_mid),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

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
                draw_hand_landmarks_with_measurements(image, hand_landmarks)
        
        if face_censored:
            if face_results.detections:
                last_face_detection = face_results
                censor_face(image, face_results)
            elif last_face_detection is not None:
                censor_face(image, last_face_detection)
        
        cv2.imshow(':P', image)
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cam.send(frame_rgb)
        cam.sleep_until_next_frame()
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            face_censored = not face_censored
        elif key == ord('m'):
            show_measurements = not show_measurements

    cap.release()
    cv2.destroyAllWindows() 