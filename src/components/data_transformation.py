import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataTransformationConfig: 
    ## preprocessor_obj_file_path: str = "artifacts/preprocessor_obj.pkl" # preprocessor object file path
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor_obj.pkl') # preprocessor object file path

class DataTransformation:
    def __init__ (self): # this init function is called when the class is instantiated. it is the constructor of the class
        self.data_transforamtion_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_features = ['writing_score', 'reading_score']
            categorical_features = ['gender', 'ethinicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]) # this is a pipeline for numerical features which first imputes the missing values with the median value and then scales the numerical features

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]) # this is a pipeline for categorical features which first imputes the missing values with the most frequent value and then one hot encodes the categorical features
            
        except Exception as e:
            logging.error("Error while getting data transformer object")
            raise CustomException(e, sys.exc_info())