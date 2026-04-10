
import pandas as pd
from flask import Flask, request, render_template
import pickle
import numpy as np
app = Flask(__name__)

# Load model (make sure you saved the full pipeline, not just the raw SVC)
with open("LG_B_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')  # HTML template with links to Predictions

# Predictions route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        expected_features = [
            'Pregnancies',
            'Glucose',
            'BloodPressure',
            'SkinThickness',
            'Insulin',
            'BMI',
            'DiabetesPedigreeFunction',
            'Age'
            ]

# Collect features from form
        features = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age']),
            ]

# Wrap in DataFrame with column names
        input_df = pd.DataFrame([features], columns=expected_features)

# Predict
        prediction = model.predict(input_df)

        labels = {0: "Not diabetes patient", 1: "Diabetes Patient"}
        return f"Predicted diabetes status: {labels[prediction[0]]}"


    # If GET request, show the form
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
