import cv2
import mediapipe as mp
import numpy as np
import tflite_runtime.interpreter as tflite
import time

# Load model
interpreter = tflite.Interpreter(model_path="custom_asl_model.tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# Load labels + autocomplete word list
labels = np.load("label_classes.npy", allow_pickle=True)
with open("words.txt", "r") as f:
    WORD_LIST = [line.strip() for line in f]

# Autocomplete
def get_suggestions(current, word_list, max_suggestions=3):
    current = current.lower()
    return [word for word in word_list if word.startswith(current)][:max_suggestions]

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# State variables
current_word = ""
transcript = ""
prediction_history = []
HISTORY_LENGTH = 10
last_added_time = 0
COOLDOWN_SECONDS = 2.0
suggestions = []

# Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

            if len(landmarks) == 63:
                input_data = np.array(landmarks, dtype=np.float32).reshape(1, -1)
                interpreter.set_tensor(input_index, input_data)
                interpreter.invoke()
                output = interpreter.get_tensor(output_index)[0]

                pred_index = int(np.argmax(output))
                confidence = float(output[pred_index])
                predicted_letter = labels[pred_index]

                # Stable prediction history
                prediction_history.append(predicted_letter)
                if len(prediction_history) > HISTORY_LENGTH:
                    prediction_history.pop(0)

                if (
                    prediction_history.count(predicted_letter) == HISTORY_LENGTH and
                    time.time() - last_added_time > COOLDOWN_SECONDS
                ):
                    current_word += predicted_letter
                    last_added_time = time.time()
                    prediction_history.clear()
                    suggestions = get_suggestions(current_word, WORD_LIST)

                cv2.putText(frame, f"{predicted_letter} ({confidence:.2f})", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    else:
        # Still update suggestions if no hand is detected
        suggestions = get_suggestions(current_word, WORD_LIST)

    # Persistent UI overlays
    suggestion_y = 80
    for i, s in enumerate(suggestions):
        cv2.putText(frame, f"{i+1}. {s}", (10, suggestion_y + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 255, 150), 2)

    cv2.rectangle(frame, (0, frame.shape[0] - 120), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.putText(frame, f"Word: {current_word}", (10, frame.shape[0] - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(frame, f"Transcript: {transcript}", (10, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)

    cv2.imshow("ASL Translator with Autocomplete", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        transcript += current_word + " "
        current_word = ""
        suggestions = []

cap.release()
cv2.destroyAllWindows()
