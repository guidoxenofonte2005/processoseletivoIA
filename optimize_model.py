import tensorflow as tf
import os

# carregamento inicial do modelo
model = tf.keras.models.load_model("model.h5")

# conversão do modelo
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# quantização
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# salvamento
tflite_model_path = "model.tflite"
open(tflite_model_path, "wb").write(tflite_model)
