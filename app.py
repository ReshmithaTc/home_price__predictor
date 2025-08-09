import pickle
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and the columns list
try:
    with open('model.pkl', 'rb') as file:
        model_data = pickle.load(file)
        model = model_data['model']
        model_columns = model_data['columns']
except FileNotFoundError:
    model = None
    model_columns = None
    print("Warning: model.pkl not found. Please run train_model.py first.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not available. Please run train_model.py.'})

    try:
        # Get data from the form
        square_feet = float(request.form.get('square_feet'))
        bedrooms = float(request.form.get('bedrooms'))
        toilets = float(request.form.get('toilets'))
        age = float(request.form.get('age'))
        city = request.form.get('city')

        garage = 1.0 if 'garage' in request.form else 0.0
        swimming_pool = 1.0 if 'swimming_pool' in request.form else 0.0
        fireplace = 1.0 if 'fireplace' in request.form else 0.0
        garden = 1.0 if 'garden' in request.form else 0.0

        # Create a new DataFrame with a row of zeros, matching the model's column order
        new_data = pd.DataFrame(0, index=[0], columns=model_columns)
        
        # Populate the DataFrame with the user's input
        new_data['square_feet'] = square_feet
        new_data['bedrooms'] = bedrooms
        new_data['toilets'] = toilets
        new_data['age'] = age
        new_data['garage'] = garage
        new_data['swimming_pool'] = swimming_pool
        new_data['fireplace'] = fireplace
        new_data['garden'] = garden

        # Handle the one-hot encoded 'city' feature
        if f'city_{city}' in model_columns:
            new_data[f'city_{city}'] = 1.0

        # Make the prediction
        prediction = model.predict(new_data)
        predicted_price = int(prediction[0])

        return jsonify({'prediction': f'${predicted_price:,.0f}'})

    except (ValueError, KeyError) as e:
        return jsonify({'error': f'Invalid input: {e}'})

if __name__ == '__main__':
    app.run(debug=True)