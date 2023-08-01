from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

application = Flask(__name__) # this is the name of the module/package that is calling this app. It is used so that Flask knows where to look for templates, static files, and so on.
# route for the application home page
@application.route('/')
def home():
    return render_template('index.html')

# route for the application predict page
@application.route('/predict', methods=['GET','POST'])
def predict():
    try:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethinicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=request.form.get('reading_score'),
            writing_score=request.form.get('writing_score')
        )
        prediction_df=pd.DataFrame([vars(data)])
        logging.info(prediction_df)
        prediction_pipeline=PredictPipeline()
        prediction=prediction_pipeline.predict(prediction_df)
        logging.info("Your predicted score is {prediction}")
        return render_template('home.html', results='Your predicted score is {}'.format(prediction))

    except Exception as e:
        return jsonify({'error': str(e)})