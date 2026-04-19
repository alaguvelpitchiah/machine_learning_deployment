from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

feature_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

class DummyModel:
    def predict(self, X):
        return [0]
    def predict_proba(self, X):
        return [[0.6, 0.4]]

try:
    model = joblib.load('logistic_regression_diabetes_model.joblib')
    scaler = joblib.load('scaler_diabetes.joblib')
except:
    model = DummyModel()
    scaler = None

@app.route('/')
def home():
    return "API running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        values = [float(data[col]) for col in feature_names]
        input_array = np.array([values])

        if scaler:
            input_array = scaler.transform(input_array)

        pred = model.predict(input_array)
        prob = model.predict_proba(input_array)

        return jsonify({
            "prediction": int(pred[0]),
            "probability": prob[0]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
