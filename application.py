from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and scaler
try:
    model = joblib.load('logistic_regression_diabetes_model.joblib')
    scaler = joblib.load('scaler_diabetes.joblib')
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

feature_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# Health check route
@app.route('/')
def home():
    return "Flask API is running"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded'}), 500

    try:
        data = request.get_json(force=True)

        # Validate input fields
        for col in feature_names:
            if col not in data:
                return jsonify({'error': f'Missing field: {col}'}), 400

        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        input_df = input_df[feature_names]

        # Convert types
        input_df = input_df.astype(float)

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        return jsonify({
            'prediction': int(prediction[0]),
            'probability_class_0': float(prediction_proba[0][0]),
            'probability_class_1': float(prediction_proba[0][1])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
