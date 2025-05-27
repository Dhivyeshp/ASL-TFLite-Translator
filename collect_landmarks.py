import cv2
import mediapipe as mp
import csv
import os
import time

label = input("Enter the label for this gesture (e.g. A, B, C): ")

folder = f"data/{label}"
os.makedirs(folder, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
sample_id = 0

print("Starting collection in 3 seconds...")
time.sleep(3)

while sample_id < 200:
    ret, frame = cap.read()
    if not ret:
        continue

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            # Save to CSV
            filename = os.path.join(folder, f"{label}_{sample_id}.csv")
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(data)

            sample_id += 1
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Samples collected: {sample_id}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Collecting Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
