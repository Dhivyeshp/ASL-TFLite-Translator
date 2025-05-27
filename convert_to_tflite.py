import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("asl_model.keras")

# TFLite conversion setup
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS
]

# Try converting
try:
    tflite_model = converter.convert()
    with open("custom_asl_model.tflite", "wb") as f:
        f.write(tflite_model)
    print("✅ Model successfully converted to custom_asl_model.tflite")
except Exception as e:
    print("❌ Conversion failed:", e)
