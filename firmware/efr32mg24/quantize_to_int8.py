import tensorflow as tf

# Load the FP32 model
converter = tf.lite.TFLiteConverter.from_saved_model("tf_saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]

# Convert to INT8
tflite_model = converter.convert()

# Save it
with open("phinet_crnn_int8.tflite", "wb") as f:
    f.write(tflite_model)
