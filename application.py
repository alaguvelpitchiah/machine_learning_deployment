from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# -----------------------------
# Dummy fallback model
# -----------------------------
class DummyModel:
    def predict(self, X):
        return [0]

    def predict_proba(self, X):
        return [[0.6, 0.4]]

model = None
scaler = None

# -----------------------------
# Load model safely
# -----------------------------
try:
    model = joblib.load('logistic_regression_diabetes_model.joblib')
    scaler = joblib.load('scaler_diabetes.joblib')
    print("✅ Model loaded")
except Exception as e:
    print("❌ Model not found, using dummy model:", e)
    model = DummyModel()
    scaler = None

feature_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

@app.route('/')
def home():
    return jsonify({"status": "API running"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No input data'}), 400

        missing = [f for f in feature_names if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        df = pd.DataFrame([data])[feature_names]
        df = df.astype(float)

        if scaler:
            df = scaler.transform(df)

        pred = model.predict(df)
        prob = model.predict_proba(df)

        return jsonify({
            "prediction": int(pred[0]),
            "probability": prob[0]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
