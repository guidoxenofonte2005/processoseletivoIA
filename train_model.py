import keras
from keras import layers, models
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_value, y_train, y_value = train_test_split(x_train, y_train, test_size=0.25)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation="relu"))

model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(64, (3, 3), activation="relu"))

model.add(layers.Flatten())

model.add(layers.Dense(28, activation="relu"))
model.add(layers.Dense(16, activation="softmax"))

optimizer = Adam(learning_rate=0.01)

model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
