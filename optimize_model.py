import time

import numpy as np
import tensorflow as tf
import os

import keras

from sklearn.metrics import classification_report
from ai_edge_litert.interpreter import Interpreter

# carregamento inicial do modelo
model = tf.keras.models.load_model("model.h5")

# carregamento dos dados
(x_train, _), (_, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)


# obtenção do dataset representativo
def get_representative_dataset():
    for i in range(100):
        sample = x_train[i : i + 1]
        yield [sample]


# conversão do modelo
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# quantização
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# uso do dataset representativo
converter.representative_dataset = get_representative_dataset

# fallback para caso não tenha suporte a int8
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,
]

# quantização de entradas e saídas para int8
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# conversão do modelo para tflite
tflite_model = converter.convert()

# salvamento
tflite_model_path = "model.tflite"
open(tflite_model_path, "wb").write(tflite_model)


# AVALIAÇÃO DO MODELO .H5 (KERAS)
# carregamento e ajuste do dataset de teste
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

start_time = time.time()
y_prediction_keras = model.predict(x_test).argmax(axis=1)
keras_latency = (time.time() - start_time) / len(x_test) * 1000 # cálculo em milisegundos

print("\033[1;31m=== REPORT 1: MODELO KERAS (.h5) ===\033[0m")
print(classification_report(y_test, y_prediction_keras))


# AVALIAÇÃO DO MODELO .TFLITE
# configuração do interpretador do TFLite
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# detalhes de entrada e saída
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# escala e zero_point são usados para converter de float32 para int8
input_scale, input_zero_point = input_details[0]['quantization']

y_prediction_tflite = []
start_time = time.time()

for i in range(len(x_test)):
    # converte a amostra float32 para int8 usando escala e zero_point
    sample = x_test[i:i+1] / input_scale + input_zero_point
    sample = sample.astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    y_prediction_tflite.append(np.argmax(output))

tflite_latency = (time.time() - start_time) / len(x_test) * 1000
y_pred_tflite = np.array(y_prediction_tflite)

print("\033[1;31m=== REPORT 2: MODELO TFLITE (.tflite) ===\033[0m")
print(classification_report(y_test, y_pred_tflite))

# COMPARATIVO DE TAMANHO
keras_size = os.path.getsize("model.h5") / 1024
tflite_size = os.path.getsize("model.tflite") / 1024
keras_acc = np.mean(y_prediction_keras == y_test) * 100
tflite_acc = np.mean(y_pred_tflite == y_test) * 100

print("\033[1;31m=== COMPARATIVO GERAL: TAMANHO, LATÊNCIA E ACCURACY ===\033[0m")
print(f"{'Métrica':<20} {'Keras (.h5)':>15} {'TFLite':>15} {'Diferença':>15}")
print(f"{'Accuracy':<20} {keras_acc:>14.2f}% {tflite_acc:>14.2f}% {(tflite_acc - keras_acc):>+14.2f}%")
print(f"{'Tamanho':<20} {keras_size:>13.1f}KB {tflite_size:>13.1f}KB {((tflite_size - keras_size) / keras_size * 100):>+13.1f}%")
print(f"{'Latência média':<20} {keras_latency:>13.3f}ms {tflite_latency:>13.3f}ms {((tflite_latency - keras_latency) / keras_latency * 100):>+13.1f}%")