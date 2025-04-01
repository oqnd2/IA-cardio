from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd

# Iniciar la aplicación Flask
app = Flask(__name__)

# Cargar el dataset (solo para obtener el mapeo de recomendaciones)
dataset = pd.read_csv("dataset.csv")

# Mapear recomendaciones a índices
recommendation_to_index = {rec: i for i, rec in enumerate(dataset["recommendation"].unique())}
index_to_recommendation = {i: rec for rec, i in recommendation_to_index.items()}

# Cargar el modelo entrenado en lugar de reentrenarlo cada vez
model = tf.keras.models.load_model("modelo_recomendaciones.h5")

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
