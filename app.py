from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd

app = Flask(__name__)

try:
    print("📂 Cargando el dataset...")
    dataset = pd.read_csv("dataset.csv")
    print("✅ Dataset cargado con éxito")

    recommendation_to_index = {rec: i for i, rec in enumerate(dataset["recommendation"].unique())}
    index_to_recommendation = {i: rec for rec, i in recommendation_to_index.items()}

    print("📂 Cargando modelo...")
    model = tf.keras.models.load_model("modelo_recomendaciones.h5")
    print("✅ Modelo cargado con éxito")

except Exception as e:
    print(f"❌ ERROR al cargar el modelo: {e}")

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.json
        bpm_value = np.array([[data["bpm"]]])  # Convertir a array para el modelo

        prediction = model.predict(bpm_value)
        recommendation_index = np.argmax(prediction)
        recommendation_text = index_to_recommendation[recommendation_index]

        return jsonify({"bpm": data["bpm"], "recommendation": recommendation_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
