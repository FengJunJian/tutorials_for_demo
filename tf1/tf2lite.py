import tensorflow as tf
import os
saved_model_dir='iris_model1/'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter
tflite_model = converter.convert()
with open(os.path.join(saved_model_dir,"converted_model.tflite"), "wb") as f:
    f.write(tflite_model)