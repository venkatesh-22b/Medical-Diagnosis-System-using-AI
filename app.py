from flask import Flask, render_template, request, jsonify
import pickle
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load all models
def load_models():
    models = {}
    model_files = {
        'diabetes': 'Models/diabetes_model.sav',
        'heart_disease': 'Models/heart_disease_model.sav',
        'parkinsons': 'Models/parkinsons_model.sav',
        'lung_cancer': 'Models/lungs_disease_model.sav',
        'thyroid': 'Models/Thyroid_model.sav'
    }
    
    for name, path in model_files.items():
        try:
            models[name] = pickle.load(open(path, 'rb'))
        except Exception as e:
            print(f"Error loading {name} model: {str(e)}")
            models[name] = None
    
    return models

MODELS = load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/<disease>', methods=['POST'])
def predict(disease):
    try:
        data = request.json
        
        if disease == 'diabetes':
            features = [
                float(data['pregnancies']), float(data['glucose']), 
                float(data['bloodPressure']), float(data['skinThickness']),
                float(data['insulin']), float(data['bmi']),
                float(data['diabetesPedigree']), float(data['age'])
            ]
            
        elif disease == 'heart':
            features = [
                float(data['age']), float(data['sex']), float(data['cp']),
                float(data['trestbps']), float(data['chol']), float(data['fbs']),
                float(data['restecg']), float(data['thalach']), float(data['exang']),
                float(data['oldpeak']), float(data['slope']), float(data['ca']),
                float(data['thal'])
            ]
            
        elif disease == 'parkinsons':
            features = [
                float(data[key]) for key in [
                    'fo', 'fhi', 'flo', 'jitter_percent', 'jitter_abs',
                    'rap', 'ppq', 'ddp', 'shimmer', 'shimmer_db',
                    'apq3', 'apq5', 'apq', 'dda', 'nhr', 'hnr',
                    'rpde', 'dfa', 'spread1', 'spread2', 'd2', 'ppe'
                ]
            ]
            
        elif disease == 'lung_cancer':
            features = [
                float(data[key]) for key in [
                    'gender', 'age', 'smoking', 'yellow_fingers',
                    'anxiety', 'peer_pressure', 'chronic_disease',
                    'fatigue', 'allergy', 'wheezing', 'alcohol',
                    'coughing', 'shortness_breath', 'swallowing_difficulty',
                    'chest_pain'
                ]
            ]
            
        elif disease == 'thyroid':
            features = [
                float(data['age']), float(data['sex']),
                float(data['on_thyroxine']), float(data['tsh']),
                float(data['t3_measured']), float(data['t3']),
                float(data['tt4'])
            ]
        
        # Make prediction
        model = MODELS[disease]
        if model is None:
            return jsonify({'error': f'Model for {disease} not loaded properly'})
            
        prediction = model.predict([features])[0]
        
        return jsonify({
            'prediction': int(prediction),
            'message': 'Positive' if prediction == 1 else 'Negative'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 