from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os
import sys
import importlib.metadata

app = Flask(__name__)
CORS(app)

def check_versions():
    """Check and log package versions for debugging"""
    packages = ['pandas', 'numpy', 'scikit-learn', 'flask']
    versions = {}
    
    for package in packages:
        try:
            version = importlib.metadata.version(package)
            versions[package] = version
            print(f"✅ {package}: {version}")
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "Not found"
            print(f"❌ {package}: Not installed")
    
    return versions

print("🔍 Checking package versions...")
versions = check_versions()

# Load model with better error handling
try:
    print("📦 Loading model...")
    with open('best_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        model_name = model_data['model_name']
        print(f"✅ Model loaded: {model_name}")
        
    # Test prediction to verify model works
    test_data = pd.DataFrame([{
        'Gender': 1,
        'Age': 45,
        'Annual Salary': 70000,
        'Credit Card Debt': 10000,
        'Net Worth': 400000
    }])
    
    test_prediction = model.predict(test_data)[0]
    print(f"🧪 Test prediction: ${test_prediction:,.2f}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    model_name = "No model"

@app.route('/')
def home():
    return jsonify({
        'message': 'Car Price Predictor API - Render Deployment',
        'status': 'active',
        'model_loaded': model_name,
        'versions': versions,
        'endpoints': {
            '/health': 'GET - API status',
            '/predict': 'POST - Predict car price',
            '/debug': 'GET - Debug info'
        }
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy' if model is not None else 'no model',
        'model': model_name,
        'versions': versions
    })

@app.route('/debug')
def debug():
    """Debug endpoint to check system info"""
    return jsonify({
        'python_version': sys.version,
        'working_directory': os.getcwd(),
        'files_in_directory': os.listdir('.'),
        'versions': versions,
        'model_status': 'loaded' if model is not None else 'not loaded'
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not available'}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = ['Gender', 'Age', 'Annual Salary', 'Credit Card Debt', 'Net Worth']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({'error': f'Missing fields: {missing_fields}'}), 400
        
        # Prepare input data
        input_data = pd.DataFrame([{
            'Gender': int(data['Gender']),
            'Age': float(data['Age']),
            'Annual Salary': float(data['Annual Salary']),
            'Credit Card Debt': float(data['Credit Card Debt']),
            'Net Worth': float(data['Net Worth'])
        }])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            'predicted_car_price': round(prediction, 2),
            'model_used': model_name,
            'input_data': data,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f"🚀 Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
