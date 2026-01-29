import cv2
import numpy as np
import time
import ai_edge_litert.interpreter as litert

# 1. SETUP & HARDWARE ACCELERATION
MODEL_PATH = "models/yolo11n-pose_int8.tflite"
interpreter = litert.Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_size = input_details[0]['shape'][1]

# 2. JUMPING JACK LOGIC VARIABLES
count = 0
stage = "down"  # Stages: "up" or "down"

def calculate_angle(a, b, c):
    """Calculates angle between three keypoints (e.g., hip, shoulder, elbow)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

# 3. CAMERA SETUP
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
prev_time = time.time()

print("Jumping Jack Counter Active!")
while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # FAST PRE-PROCESSING
    img = cv2.resize(frame, (input_size, input_size))
    # Note: Using float32 for the 'entry door' as required by the model
    input_tensor = img.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # INFERENCE
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    
    # OUTPUT PROCESSING
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    output = output.T # Transpose to [2100, 56]
    
    # Grab the highest confidence detection
    best_idx = np.argmax(output[:, 4])
    if output[best_idx, 4] > 0.4:
        kpts = output[best_idx, 5:].reshape(17, 3)
        h, w, _ = frame.shape

        # GET KEYPOINTS (Normalized to pixel values)
        l_shoulder = [kpts[5][0] * w, kpts[5][1] * h]
        r_shoulder = [kpts[6][0] * w, kpts[6][1] * h]
        l_hip = [kpts[11][0] * w, kpts[11][1] * h]
        r_hip = [kpts[12][0] * w, kpts[12][1] * h]
        l_elbow = [kpts[7][0] * w, kpts[7][1] * h]
        r_elbow = [kpts[8][0] * w, kpts[8][1] * h]

        # CALCULATE ANGLES
        # Angle between hip, shoulder, and elbow (Armpit angle)
        l_angle = calculate_angle(l_hip, l_shoulder, l_elbow)
        r_angle = calculate_angle(r_hip, r_shoulder, r_elbow)

        # JUMPING JACK LOGIC
        # If arms are above shoulders (angle > 140) and were previously down
        if l_angle > 140 and r_angle > 140:
            stage = "up"
        # If arms come back down (angle < 50) and were previously up, count a rep
        if l_angle < 50 and r_angle < 50 and stage == "up":
            stage = "down"
            count += 1
            print(f"Rep counted! Total: {count}")

    # UI & FPS
    fps = 1 / (time.time() - prev_time)
    prev_time = time.time()
    
    # HUD
    cv2.rectangle(frame, (0,0), (250, 100), (0,0,0), -1)
    cv2.putText(frame, f"REPS: {count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
    
    cv2.imshow("Pi Jumping Jack Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()