from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd

# Cargar el dataset
dataset = pd.read_csv("dataset.csv")

# Crear el modelo de la red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(1,)),  # Capa oculta
    tf.keras.layers.Dense(16, activation="relu"),  # Segunda capa oculta
    tf.keras.layers.Dense(len(dataset["recommendation"].unique()), activation="softmax")  # Capa de salida
])

# Compilar el modelo
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Mapear recomendaciones a índices
recommendation_to_index = {rec: i for i, rec in enumerate(dataset["recommendation"].unique())}
index_to_recommendation = {i: rec for rec, i in recommendation_to_index.items()}

# Convertir los BPM y las recomendaciones a valores numéricos
dataset["recommendation_index"] = dataset["recommendation"].map(recommendation_to_index)

# Entrenar el modelo
X = dataset["bpm"].values.reshape(-1, 1)  # Entrada: BPMs
y = dataset["recommendation_index"].values  # Salida: Índice de la recomendación

model.fit(X, y, epochs=100, verbose=1)

# Iniciar el servidor Flask
app = Flask(__name__)

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.json
        bpm_value = np.array([[data["bpm"]]])  # Convertir a array para el modelo

        prediction = model.predict(bpm_value)
        recommendation_index = np.argmax(prediction)  # Obtener la recomendación más probable
        recommendation_text = index_to_recommendation[recommendation_index]

        return jsonify({"bpm": data["bpm"], "recommendation": recommendation_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
