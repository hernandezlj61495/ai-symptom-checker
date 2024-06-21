from flask import render_template, request, jsonify
from . import create_app
import joblib

app = create_app()

# Load the pre-trained model
model = joblib.load('models/symptom_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['symptoms']
    symptoms = [data]  # The model expects a 2D array
    prediction = model.predict(symptoms)
    recommendations = "See a doctor" if prediction[0] == 1 else "Home care"
    return jsonify({'prediction': prediction[0], 'recommendations': recommendations})
