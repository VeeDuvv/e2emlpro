from flask import Flask, request, jsonify, render_template
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

application = Flask(__name__) # this is the name of the module/package that is calling this app. It is used so that Flask knows where to look for templates, static files, and so on.
app=application

# route for the application home page
@application.route('/')
def home():
    logging.info("Calling home page")
    return render_template('index.html')

# route for the application predict page
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else: 
        try:
            input_data=CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethinicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=request.form.get('reading_score'),
                writing_score=request.form.get('writing_score')
            )
            input_data_df=input_data.get_data_as_data_frame()
            logging.info(input_data_df)
            prediction_pipeline=PredictPipeline()
            prediction=prediction_pipeline.predict(input_data_df)
            logging.info("Your predicted score is {prediction}")
            return render_template('home.html', results='Your predicted score is {}'.format(prediction))

        except Exception as e:
            logging.error("Error while initiating flask application")
            raise CustomException(e, sys.exc_info())
    
if __name__=="__main__":
    app.run(host="0.0.0.0")  