import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load and prepare the dataset
df = pd.read_csv("asl_dataset.csv", header=None)
X = df.iloc[:, :-1].astype(np.float32)
y = df.iloc[:, -1]

# Encode labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save label mappings
np.save("label_classes.npy", le.classes_)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build a simple MLP model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Save model
model.save("asl_model.keras")
