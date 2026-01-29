import cv2
import numpy as np
import ai_edge_litert.interpreter as litert

# 1. Setup the Interpreter
# Change this path to your specific pruned/slimmed TFLite model
MODEL_PATH = "models/yolo11n-pose_int8.tflite"
interpreter = litert.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'] # Usually [1, 320, 320, 3] or [1, 640, 640, 3]

# 2. Setup USB Webcam
cap = cv2.VideoCapture(0)

print("Starting Inference...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Pre-process: Resize and Normalize
    # Match the image size used during your export (e.g., 320x320)
    input_img = cv2.resize(frame, (input_shape[1], input_shape[2]))
    input_img = input_img.astype(np.float32) / 255.0  # Normalize to [0,1]
    input_img = np.expand_dims(input_img, axis=0)     # Add batch dimension

    # 3. Run Inference
    interpreter.set_tensor(input_details[0]['index'], input_img)
    interpreter.invoke()

    # 4. Get Results (Keypoints)
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Process output_data here to find joints (elbows, knees, etc.)
    # and add your jumping jack counting logic!

    cv2.imshow('Pi Jumping Jack Counter', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()