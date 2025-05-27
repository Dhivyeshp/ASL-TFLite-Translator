import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tflite_runtime.interpreter as tflite

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# ASL A–Y (excluding J and Z which require motion)
labels = list("ABCDEFGHIKLMNOPQRSTUVWXY")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Text-to-Speech
engine = pyttsx3.init()
spoken_last = ""

# Webcam Setup
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 21 landmarks × (x, y, z) = 63 features
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            if len(data) == 63:
                # Pad to match model input shape (276 features)
                padded_data = data + [0.0] * (276 - len(data))
                input_data = np.array(padded_data, dtype=np.float32).reshape(1, -1)

                interpreter.set_tensor(input_index, input_data)
                interpreter.invoke()
                output = interpreter.get_tensor(output_index)[0]

                pred_index = int(np.argmax(output))
                confidence = float(output[pred_index])
                predicted_letter = labels[pred_index]

                # Show predicted letter on screen
                cv2.putText(frame, f'{predicted_letter} ({confidence:.2f})', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

                print(f"Prediction: {predicted_letter} | Confidence: {confidence:.2f}")

                # Only speak if confident and not a repeat
                if confidence > 0.8 and predicted_letter != spoken_last:
                    engine.say(predicted_letter)
                    engine.runAndWait()
                    spoken_last = predicted_letter

    cv2.imshow("ASL Translator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
