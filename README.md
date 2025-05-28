# 🤟 ASL-TFLite Translator

A real-time **American Sign Language to Text translator** that runs fully offline on a **Raspberry Pi 4B** using **MediaPipe**, **TensorFlow Lite**, and a custom-trained neural network.

## 🧠 How It Works
- **MediaPipe** extracts 21 3D landmarks from the hand in each frame
- **Landmarks (63D vector)** are passed to a trained TFLite model
- **Predicted letter** is shown with live autocomplete suggestions
- All computation runs locally on-device (no internet required)

---

## 🏗️ Project Structure
```
ASL-TFLite-Translator/
├── asl_translator_custom.py     # Real-time inference on Pi
## 📸 Demo

Real-time hand gesture → letter prediction → autocomplete suggestions.

## 📸 Demo

Real-time hand gesture → letter prediction → autocomplete suggestions.

---
---
## 📸 Demo

Real-time hand gesture → letter prediction → autocomplete suggestions.

---
## 📸 Demo

Real-time hand gesture → letter prediction → autocomplete suggestions.

---
├── train_asl_model.py           # Keras model training
├── convert_to_tflite.py         # Convert to TensorFlow Lite
├── label_classes.npy            # Class label mappings
├── model/
│   └── custom_asl_model.tflite
└── data/                        # (optional) Collected gesture data
```

---

## 🚀 Getting Started
### 🧰 Requirements
```bash
pip3 install -r requirements.txt
```

### 🐍 Run the Translator
```bash
python3 asl_translator_custom.py
```

### 📦 Train Your Own Model
```bash
python3 train_asl_model.py
python3 convert_to_tflite.py
```
Make sure you update the label set and number of classes accordingly.

---

## 🔍 Tech Stack
- Python 3.11
- OpenCV + NumPy
- MediaPipe (Hands)
- TensorFlow / TensorFlow Lite
- Raspberry Pi 4B (or any Linux SBC)

---

## 📈 Future Improvements
- Better temporal smoothing (LSTM or moving window)
- Word/phrase prediction (not just letters)
- Text-to-speech integration
- Mobile or web UI version

---

## 📜 License
[MIT](LICENSE)

---

## 🙌 Credits
Huge thanks to:
- [MediaPipe](https://google.github.io/mediapipe/)
- TensorFlow Lite team
- Community examples + ASL datasets

---

> Built with ❤️ to make communication more accessible.

---

## 🔧 Python Scripts

### `asl_translator_custom.py`
```python
import cv2
import numpy as np
import mediapipe as mp
import tensorflow.lite as tflite

interpreter = tflite.Interpreter(model_path="model/custom_asl_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]['index']
output_index = output_details[0]['index']
labels = np.load("label_classes.npy", allow_pickle=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)
prev_letter = ''
cooldown = 15
frame_delay = 0
transcript = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            input_data = np.array(coords, dtype=np.float32).reshape(1, -1)
            interpreter.set_tensor(input_index, input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_index)
            pred_index = np.argmax(output[0])
            confidence = output[0][pred_index]

            if confidence > 0.8:
                predicted_letter = labels[pred_index]
                if predicted_letter != prev_letter:
                    transcript += predicted_letter
                    prev_letter = predicted_letter
                    frame_delay = cooldown
                elif frame_delay > 0:
                    frame_delay -= 1
            else:
                prev_letter = ''
    else:
        prev_letter = ''

    cv2.putText(frame, f"{transcript}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("ASL Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

> Let me know if you'd like me to add the `train_asl_model.py` and `convert_to_tflite.py` files next!
