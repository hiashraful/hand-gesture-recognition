import cv2
import mediapipe as mp
import asyncio
import websockets
import json
import threading
import time
import numpy as np
import csv
import copy
import itertools
from collections import deque, Counter
import tensorflow as tf

# Add these new imports for the ML-based gesture detection
class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = np.argmax(np.squeeze(result))
        return result_index

class PointHistoryClassifier(object):
    def __init__(
        self,
        model_path='model/point_history_classifier/point_history_classifier.tflite',
        score_th=0.5,
        invalid_value=0,
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.score_th = score_th
        self.invalid_value = invalid_value

    def __call__(self, point_history_list):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([point_history_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = np.argmax(np.squeeze(result))

        if np.squeeze(result)[result_index] < self.score_th:
            result_index = self.invalid_value

        return result_index

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    model_complexity=1,
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.7
)

pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.7
)

# Detection parameters
MIN_DISTANCE = 0.5  # meters
MAX_DISTANCE = 2.0  # meters
CENTER_THRESHOLD = 0.3  # 30% from center
COOLDOWN_TIME = 10  # seconds

# Camera parameters
FOCAL_LENGTH = 1000  # pixels
KNOWN_PERSON_HEIGHT = 1.7  # meters

# Initialize ML classifiers
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

# Load gesture labels
def load_labels():
    """Load gesture labels from CSV files"""
    keypoint_classifier_labels = []
    try:
        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
    except FileNotFoundError:
        # Default labels if file doesn't exist
        keypoint_classifier_labels = ['👍 Thumbs Up', '✌️ Peace Sign', '🖕 Middle Finger', 
                                    '🤘 Heavy Metal', '🤟 Our Sign']
    
    point_history_classifier_labels = []
    try:
        with open('model/point_history_classifier/point_history_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            point_history_classifier_labels = [row[0] for row in csv.reader(f)]
    except FileNotFoundError:
        point_history_classifier_labels = ['Stop', 'Clockwise', 'Counter Clockwise', 'Move']
    
    return keypoint_classifier_labels, point_history_classifier_labels

keypoint_classifier_labels, point_history_classifier_labels = load_labels()

# Coordinate history for dynamic gestures
history_length = 16
point_history = deque(maxlen=history_length)
finger_gesture_history = deque(maxlen=history_length)

current_data = {
    "person_count": 0,
    "gesture": "🤷 Unknown",
    "timestamp": time.time(),
    "status": "running",
    "detection_triggered": False,
    "last_trigger_time": 0
}

connected_clients = set()
data_lock = threading.Lock()

def calc_landmark_list(image, landmarks):
    """Calculate landmark list from MediaPipe hand landmarks"""
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    """Preprocess landmark coordinates for ML model"""
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def pre_process_point_history(image, point_history):
    """Preprocess point history for dynamic gesture recognition"""
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

    return temp_point_history

def estimate_distance(person_height_pixels, frame_height):
    """Estimate distance using person height in pixels"""
    if person_height_pixels <= 0:
        return None
    
    distance = (KNOWN_PERSON_HEIGHT * FOCAL_LENGTH) / person_height_pixels
    return distance

def is_person_centered(pose_landmarks, frame_width):
    """Check if person is centered in the frame"""
    if not pose_landmarks:
        return False
    
    nose = pose_landmarks.landmark[0]
    if nose.visibility < 0.5:
        return False
    
    center_x = nose.x
    center_offset = abs(center_x - 0.5)
    
    return center_offset <= CENTER_THRESHOLD

def get_person_height_pixels(pose_landmarks, frame_height):
    """Calculate person height in pixels"""
    if not pose_landmarks:
        return 0
    
    landmarks = pose_landmarks.landmark
    
    key_points = [
        (landmarks[0], "nose"),
        (landmarks[11], "left_shoulder"),
        (landmarks[12], "right_shoulder"),
        (landmarks[23], "left_hip"),
        (landmarks[24], "right_hip"),
        (landmarks[27], "left_ankle"),
        (landmarks[28], "right_ankle")
    ]
    
    visible_points = []
    for landmark, name in key_points:
        if landmark.visibility > 0.5:
            visible_points.append(landmark.y * frame_height)
    
    if len(visible_points) < 3:
        return 0
    
    height_pixels = max(visible_points) - min(visible_points)
    return height_pixels

def check_detection_conditions(pose_landmarks, frame_width, frame_height):
    """Check if person meets all detection conditions"""
    if not pose_landmarks:
        return False, None, None
    
    if not is_person_centered(pose_landmarks, frame_width):
        return False, None, None
    
    height_pixels = get_person_height_pixels(pose_landmarks, frame_height)
    if height_pixels <= 0:
        return False, None, None
    
    distance = estimate_distance(height_pixels, frame_height)
    if distance is None:
        return False, None, None
    
    in_range = MIN_DISTANCE <= distance <= MAX_DISTANCE
    
    return in_range, distance, height_pixels

def should_trigger_detection():
    """Check if enough time has passed since last detection"""
    current_time = time.time()
    time_since_last = current_time - current_data["last_trigger_time"]
    return time_since_last >= COOLDOWN_TIME

def detect_person(pose_landmarks):
    """Detect if a person is present"""
    if not pose_landmarks:
        return False
    
    key_points = [0, 2, 5, 11, 12, 13, 14, 15, 16]
    visible_count = 0
    
    for idx in key_points:
        if pose_landmarks.landmark[idx].visibility > 0.5:
            visible_count += 1
    
    return visible_count >= 5

def update_detection_data(hand_result, pose_result, frame_width, frame_height, img):
    """Update detection data with ML-based gesture recognition"""
    global current_data, point_history, finger_gesture_history
    
    with data_lock:
        person_count = 0
        gesture = "🤷 Unknown"
        detection_triggered = False
        
        # Person detection
        if pose_result.pose_landmarks:
            if detect_person(pose_result.pose_landmarks):
                person_count = 1
                
                in_range, distance, height_pixels = check_detection_conditions(
                    pose_result.pose_landmarks, frame_width, frame_height
                )
                
                if in_range and should_trigger_detection():
                    detection_triggered = True
                    current_data["last_trigger_time"] = time.time()
        
        # Hand gesture recognition using ML
        if hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:
                # Calculate landmark list
                landmark_list = calc_landmark_list(img, hand_landmarks)
                
                # Preprocess landmarks for keypoint classifier
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                
                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                # Update point history for finger gesture classification
                if hand_sign_id == 2:  # Pointing gesture (adjust based on your model)
                    point_history.append(landmark_list[8])  # Index finger tip
                else:
                    point_history.append([0, 0])
                
                # Finger gesture classification (dynamic movements)
                finger_gesture_id = 0
                if len(point_history) == history_length:
                    pre_processed_point_history_list = pre_process_point_history(img, point_history)
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
                
                # Update finger gesture history
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()
                
                # Get gesture label
                if hand_sign_id < len(keypoint_classifier_labels):
                    gesture = keypoint_classifier_labels[hand_sign_id]
                    break
        else:
            point_history.append([0, 0])
        
        current_data = {
            "person_count": person_count,
            "gesture": gesture,
            "timestamp": time.time(),
            "status": "running",
            "detection_triggered": detection_triggered,
            "last_trigger_time": current_data["last_trigger_time"]
        }

# WebSocket handling remains the same
async def websocket_handler(websocket):
    connected_clients.add(websocket)
    
    try:
        welcome_msg = {
            "type": "connection",
            "message": "Connected to person and gesture detection"
        }
        await websocket.send(json.dumps(welcome_msg))
        
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get("type") == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))
            except:
                pass
        
    except (websockets.exceptions.ConnectionClosed, 
            websockets.exceptions.InvalidMessage, 
            websockets.exceptions.ConnectionClosedError,
            EOFError):
        pass
    except Exception:
        pass
    finally:
        connected_clients.discard(websocket)

async def broadcast_data():
    while True:
        if connected_clients:
            with data_lock:
                message = json.dumps(current_data)
            
            disconnected_clients = set()
            for client in connected_clients:
                try:
                    await client.send(message)
                except:
                    disconnected_clients.add(client)
            
            for client in disconnected_clients:
                connected_clients.discard(client)
        
        await asyncio.sleep(1/30)

def start_websocket_server():
    async def server():
        server = await websockets.serve(
            websocket_handler, 
            "localhost", 
            8765,
            ping_interval=10,
            ping_timeout=5,
            max_size=2**16
        )
        
        broadcast_task = asyncio.create_task(broadcast_data())
        
        try:
            await server.wait_closed()
        except KeyboardInterrupt:
            broadcast_task.cancel()
            server.close()
            await server.wait_closed()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server())

def main():
    global current_data
    
    websocket_thread = threading.Thread(target=start_websocket_server, daemon=True)
    websocket_thread.start()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
    
    show_video = True
    
    try:
        while True:
            success, img = cap.read()
            if not success:
                break
            
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            hand_result = hands.process(img_rgb)
            pose_result = pose.process(img_rgb)
            
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            update_detection_data(hand_result, pose_result, frame_width, frame_height, img)
            
            if show_video:
                display_img = img.copy()
                h, w, c = display_img.shape
                
                # # Display detection info
                # cv2.putText(display_img, f"People: {current_data['person_count']}", (10, 30), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # cv2.putText(display_img, f"Gesture: {current_data['gesture']}", (10, 70), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # cv2.putText(display_img, f"Clients: {len(connected_clients)}", (10, 110), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Display detection conditions
                # if pose_result.pose_landmarks:
                #     in_range, distance, height_pixels = check_detection_conditions(
                #         pose_result.pose_landmarks, frame_width, frame_height
                #     )
                    
                #     if distance is not None:
                #         cv2.putText(display_img, f"Distance: {distance:.2f}m", (10, 150), 
                #                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                #     if in_range:
                #         cv2.putText(display_img, "IN RANGE", (10, 190), 
                #                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                #     else:
                #         cv2.putText(display_img, "OUT OF RANGE", (10, 190), 
                #                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                #     if current_data["detection_triggered"]:
                #         cv2.putText(display_img, "TRIGGERED!", (10, 230), 
                #                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)
                
                # Display cooldown info
                time_since_last = time.time() - current_data["last_trigger_time"]
                if time_since_last < COOLDOWN_TIME:
                    remaining = COOLDOWN_TIME - time_since_last
                #     cv2.putText(display_img, f"Cooldown: {remaining:.1f}s", (10, 270), 
                #                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                # cv2.putText(display_img, "Press 'q' to quit, 's' to hide/show video", (10, h - 20), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw hand landmarks
                if hand_result.multi_hand_landmarks:
                    for hand_landmarks in hand_result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            display_img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                        )
                
                # Draw pose landmarks
                if pose_result.pose_landmarks:
                    mp_draw.draw_landmarks(
                        display_img, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2)
                    )
                
                cv2.imshow("Enhanced Person & Gesture Detection", display_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                show_video = not show_video
                if not show_video:
                    cv2.destroyAllWindows()
    
    except KeyboardInterrupt:
        pass
    
    finally:
        with data_lock:
            current_data["status"] = "stopped"
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        import websockets
    except ImportError:
        exit(1)
    
    main()