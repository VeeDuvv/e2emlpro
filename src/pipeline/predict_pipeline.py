import sys
from src.exception import CustomException
import pandas as pd
from src.utils import load_object, evaluate_models

class PredictPipeline: 
    def __init__(self):
        self.model = None
        self.model_columns = None
        self.scaler = None
    
    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor_obj.pkl'
            model=load_object(model_path)
            preprocessor=load_object(preprocessor_path)
            data_scaled=preprocessor.transform(features)
            return model.predict(data_scaled)
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    # responsible to map all inputs from html to the model
    # these are the same inputs as our test and train data
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str, lunch: str, test_preparation_course: str, reading_score: int, writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_frame(self):
        return pd.DataFrame([vars(self)])
