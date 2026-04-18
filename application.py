from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

application = Flask(__name__)

# Load model, scaler, and feature names
MODEL_PATH = 'model.joblib'
SCALER_PATH = 'scaler.joblib'
FEATURES_PATH = 'feature_names.joblib'

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
else:
    model, scaler, feature_names = None, None, None
    print("WARNING: Model files not found. Run export_model.py first.")

@application.route('/')
def home():
    # Pass the feature names to the template to generate inputs dynamically
    return render_template('index.html', features=feature_names)

@application.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({'error': 'Model not loaded on server.'}), 500
        
    try:
        # Get data from the request
        data = request.json
        
        # Build a DataFrame with a single row, ensuring column order matches training
        input_data = {}
        for feature in feature_names:
            # If a feature is missing in the request, default to 0 (or handle error)
            input_data[feature] = [float(data.get(feature, 0.0))]
            
        df_input = pd.DataFrame(input_data)
        
        # Scale the data
        scaled_input = scaler.transform(df_input)
        
        # Predict
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1] # Prob of class 1 (M)
        
        # Format the result
        result = "Malignant (Cancerous)" if prediction == 1 else "Benign (Non-Cancerous)"
        
        return jsonify({
            'prediction': result,
            'probability': f"{probability * 100:.2f}%",
            'is_malignant': bool(prediction == 1)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    application.run(debug=True)
