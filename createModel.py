import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # Importamos el scaler

# Cargar el dataset
dataset = pd.read_csv("dataset.csv")

# Mapear recomendaciones a índices
recommendation_to_index = {rec: i for i, rec in enumerate(dataset["recommendation"].unique())}
dataset["recommendation_index"] = dataset["recommendation"].map(recommendation_to_index)

# Normalizar los BPM usando Min-Max Scaling
scaler = MinMaxScaler(feature_range=(0, 1))  # Establecemos el rango entre 0 y 1
X_scaled = scaler.fit_transform(dataset["bpm"].values.reshape(-1, 1))  # Normalizamos los BPM

# Definir el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(1,)),  # Capa oculta
    tf.keras.layers.Dense(16, activation="relu"),  # Segunda capa oculta
    tf.keras.layers.Dense(len(dataset["recommendation"].unique()), activation="softmax")  # Capa de salida
])

# Compilar el modelo
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Entrenar el modelo con los datos normalizados
y = dataset["recommendation_index"].values  # Las etiquetas no cambian
model.fit(X_scaled, y, epochs=100, verbose=1)

# Guardar el modelo entrenado
model.save("modelo_recomendaciones.h5")
print("✅ Modelo guardado exitosamente como modelo_recomendaciones.h5")
