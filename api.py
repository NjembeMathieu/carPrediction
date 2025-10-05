from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Chargement du modèle
try:
    with open('best_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        model_name = model_data['model_name']
        print(f"✅ Modèle chargé: {model_name}")
except Exception as e:
    print(f"❌ Erreur chargement modèle: {e}")
    model = None
    model_name = "Aucun modèle"

@app.route('/')
def home():
    return jsonify({
        'message': 'API Car Price Predictor - Déployée sur Render.com',
        'status': 'active',
        'endpoints': {
            '/health': 'GET - Statut API',
            '/predict': 'POST - Prédiction prix voiture'
        }
    })

@app.route('/health')
def health():
    status = 'healthy' if model is not None else 'no model'
    return jsonify({'status': status, 'model': model_name})

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if model is None:
        return jsonify({'error': 'Modèle non disponible'}), 500
    
    if request.method == 'GET':
        return jsonify({
            'message': 'Utilisez POST pour les prédictions',
            'exemple': {
                'Gender': 1,
                'Age': 45,
                'Annual Salary': 70000,
                'Credit Card Debt': 10000,
                'Net Worth': 400000
            }
        })
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Aucune donnée JSON'}), 400
        
        # Validation des champs
        required_fields = ['Gender', 'Age', 'Annual Salary', 'Credit Card Debt', 'Net Worth']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Champ manquant: {field}'}), 400
        
        # Préparation des données
        input_data = pd.DataFrame([{
            'Gender': int(data['Gender']),
            'Age': float(data['Age']),
            'Annual Salary': float(data['Annual Salary']),
            'Credit Card Debt': float(data['Credit Card Debt']),
            'Net Worth': float(data['Net Worth'])
        }])
        
        # Prédiction
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            'predicted_car_price': round(prediction, 2),
            'model_used': model_name,
            'input_data': data,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': f'Erreur: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
