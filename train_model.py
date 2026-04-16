import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()

train_x = train_x.as_type("float32") / 255
train_y = train_y.as_type("float32") / 255

train_x = train_x.reshape((train_x.shape[0], 28 * 28))
test_x = test_x.reshape((train_x.shape[0], 28 * 28))