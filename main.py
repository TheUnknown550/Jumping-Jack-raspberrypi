import cv2
import numpy as np
import time
import ai_edge_litert.interpreter as litert

# 1. Setup - Path to your INT8 model for maximum speed
MODEL_PATH = "models/yolo11n-pose_int8.tflite"
CONF_THRESHOLD = 0.25

# HARDWARE ACCELERATION: Use all 4 cores and XNNPACK
# This is the single biggest speed boost for the Raspberry Pi
interpreter = litert.Interpreter(
    model_path=MODEL_PATH,
    num_threads=4  # Utilizing all Pi cores
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_size = input_details[0]['shape'][1] 

# 2. Camera Setup - Lower resolution for faster capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 3. Connectivity Edges for Pose
EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), 
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

prev_time = time.time()

print("Optimized Inference Started...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # --- PERFORMANCE FIX 1: FAST PRE-PROCESSING ---
    # Resizing first, then handling the array.
    img = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    
    # INT8 optimization: If your model is INT8, skip the "/ 255.0" math.
    # Standard INT8 models expect 0-255 UINT8 data.
    input_tensor = np.expand_dims(img, axis=0).astype(np.uint8)

    # --- PERFORMANCE FIX 2: ACCELERATED INVOCATION ---
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    
    # 4. Get Results
    output = interpreter.get_tensor(output_details[0]['index'])[0].transpose(1, 0)

    # --- PERFORMANCE FIX 3: TARGETED POST-PROCESSING ---
    # Find only the top person to avoid looping through 2100 results
    scores = output[:, 4]
    best_idx = np.argmax(scores)
    
    if scores[best_idx] > CONF_THRESHOLD:
        det = output[best_idx]
        kpts = det[5:].reshape(17, 3) 
        h, w, _ = frame.shape

        for start, end in EDGES:
            pt1 = (int(kpts[start][0] * w), int(kpts[start][1] * h))
            pt2 = (int(kpts[end][0] * w), int(kpts[end][1] * h))
            if kpts[start][2] > 0.5 and kpts[end][2] > 0.5:
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    # FPS Display
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow("Optimized Pi Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()