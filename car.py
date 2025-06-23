from flask import Flask, render_template, request  # fixed duplicate render_template & added request import
from flask_cors import CORS, cross_origin          # import CORS and cross_origin
import pandas as pd
import numpy as np                                # missing import for numpy
import pickle

app = Flask(__name__)
cors = CORS(app)                                  # fixed CORS initialization

# Load model and data
model = pickle.load(open('pipe.pkl', 'rb'))
with open('data.pkl', 'rb') as file:
    data = pickle.load(file)

car = pd.DataFrame(data)

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = sorted(car['fuel_type'].unique())
    print("Companies list:", car_models)
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')    # name in form might be car_models (check your HTML)
    year = int(request.form.get('year'))           # convert to int
    fuel_type = request.form.get('fuel_type')
    driven = int(request.form.get('kilo_driven'))  # convert to int

    # Prepare input DataFrame with correct data types matching model training
    input_df = pd.DataFrame(
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
        data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)
    )

    prediction = model.predict(input_df)
    print(prediction)

    return str(np.round(prediction[0], 2))


if __name__ == '__main__':
    app.run(debug=True)
