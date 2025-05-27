import os
import numpy as np

X = []
y = []

data_dir = "landmarks_data"

for label in os.listdir(data_dir):
    label_path = os.path.join(data_dir, label)
    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        if file.endswith(".npy"):
            file_path = os.path.join(label_path, file)
            try:
                landmark = np.load(file_path)
                if landmark.shape == (63,):  # Ensure correct size
                    X.append(landmark)
                    y.append(label)
            except Exception as e:
                print(f"Skipping {file_path}: {e}")

X = np.array(X)
y = np.array(y)

np.save("X_landmarks.npy", X)
np.save("y_labels.npy", y)

print(f"✅ Saved dataset: {len(X)} samples across {len(set(y))} classes.")
