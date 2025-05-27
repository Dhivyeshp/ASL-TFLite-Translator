import cv2
import mediapipe as mp
import numpy as np
import os

# Ask user what label to record
label = input("Enter the letter to record (A-Z): ").upper()
assert len(label) == 1 and label.isalpha(), "Must be a single letter A-Z."

# Make directory to store data
save_dir = f"landmarks_data/{label}"
os.makedirs(save_dir, exist_ok=True)

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
count = 0

print("Recording started. Press 'q' to stop.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Flatten the landmarks to a list of 63 floats
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            np.save(f"{save_dir}/{label}_{count}.npy", np.array(landmarks))
            count += 1
            print(f"Saved sample {count} for '{label}'")

    cv2.putText(frame, f"Recording '{label}' - Count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("ASL Sample Recorder", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
