import keras
from keras import layers, models
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# CARREGAMENTO DOS DADOS DE TREINO E TESTE
# separa entre dados de treino e teste
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# retira 25% dos dados de treino para teste no treinamento
x_train, x_value, y_train, y_value = train_test_split(x_train, y_train, test_size=0.25)


# NORMALIZAÇÃO DOS DADOS
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_value = x_value.astype("float32") / 255


# AJUSTE UNIDIMENSIONAL DAS CAMADAS DE TESTE
# ajusta as imagens para 28x28 em escala de cinza
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_value = x_value.reshape(x_value.shape[0], 28, 28, 1)


# DEFINIÇÕES DO MODELO
# criação do modelo
model = models.Sequential()

# adição da camada de input
model.add(keras.layers.Input((28, 28, 1)))  # 28x28, escala de cinza

# definição das layers convolucionais
model.add(layers.Conv2D(16, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D())

# layer de dropout inicial para evitar overfitting
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(32, (3, 3), activation="relu"))
model.add(layers.Flatten())

# layer de dropout extra para evitar overfitting
model.add(layers.Dropout(0.5))

model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))


# DEFINIÇÃO DO OTIMIZADOR
optimizer = Adam(learning_rate=0.001)


# COMPILAÇÃO DO MODELO
model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",  # função de perda padrão para CNNs
    metrics=["accuracy"],
)


# DEFINIÇÃO DE EARLY STOPPING
# Técnica que para o treinamento se a métrica desejada parar de evoluir
callback = keras.callbacks.EarlyStopping(
    restore_best_weights=True, patience=1, min_delta=0.01
)


# TREINAMENTO E AVALIAÇÃO DO MODELO
print(f"\033[1;31mINICIANDO TREINAMENTO DO MODELO...\033[0m")
model.fit(
    x_train, y_train, epochs=5, validation_data=(x_value, y_value), callbacks=[callback]
)

# caso essa mensagem tenha aparecido antes das 5 epochs, significa que o 
# EarlyStopping foi ativado para evitar treinamentos desnecessários
print(f"\033[1;31mTREINAMENTO CONCLUÍDO\n\nINICIANDO AVALIAÇÃO DE MÉTRICAS...\033[0m")

y_probability = model.predict(x_test)
y_predictions = np.argmax(y_probability, axis=1)

print(classification_report(y_test, y_predictions))

print(f"\033[1;31mGERANDO MATRIZ DE CONFUSÃO...\033[0m")
conf_matrix = confusion_matrix(y_test, y_predictions)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Classe predita")
plt.ylabel("Classe real")
plt.title("Matriz de Confusão")
plt.show()


# SALVAMENTO FINAL DO MODELO
print(f"\n\033[1;31mSALVANDO MODELO EM .h5 e .keras...\n\033[0m")

model.save("model.h5")
model.save("model.keras")
