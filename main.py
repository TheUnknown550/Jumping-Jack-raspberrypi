import cv2
import numpy as np
import time
import ai_edge_litert.interpreter as litert

# 1. Setup - Path to your single best model
MODEL_PATH = "models/yolo11n-pose_int8.tflite"
CONF_THRESHOLD = 0.25

# COCO Skeleton Edges (Connections between keypoints)
EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), 
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

# Load Model
interpreter = litert.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_size = input_details[0]['shape'][1] # 320 or 640

# 2. Camera Setup
cap = cv2.VideoCapture(0)
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Pre-process
    h, w, _ = frame.shape
    img = cv2.resize(frame, (input_size, input_size))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # 3. Inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    
    # YOLO11 Output is typically [1, 56, 2100] 
    # (56 = 4 box coords + 1 class score + 17 keypoints * 3 [x,y,conf])
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    output = output.transpose(1, 0) # Change to [2100, 56]

    # 4. Post-Processing (Finding the person with highest confidence)
    # We look at the 5th value (index 4) which is the person confidence score
    scores = output[:, 4]
    best_idx = np.argmax(scores)
    
    if scores[best_idx] > CONF_THRESHOLD:
        det = output[best_idx]
        # Keypoints start at index 5. They are normalized (0-1) in YOLO11 TFLite
        kpts = det[5:].reshape(17, 3) 

        # Draw Skeleton
        for start, end in EDGES:
            pt1 = (int(kpts[start][0] * w), int(kpts[start][1] * h))
            pt2 = (int(kpts[end][0] * w), int(kpts[end][1] * h))
            if kpts[start][2] > 0.5 and kpts[end][2] > 0.5: # Confidence check
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Draw Keypoints
        for x, y, conf in kpts:
            if conf > 0.5:
                cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 0, 255), -1)

    # FPS Calculation
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now

    # Display
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Single Model Pose Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()