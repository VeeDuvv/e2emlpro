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
from src.utils import save_object

@dataclass
class DataTransformationConfig: 
    ## preprocessor_obj_file_path: str = "artifacts/preprocessor_obj.pkl" # preprocessor object file path
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor_obj.pkl') # preprocessor object file path

class DataTransformation:
    def __init__ (self): # this init function is called when the class is instantiated. it is the constructor of the class
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        '''
        This function returns the data transformer object
        '''
        try:
            num_features = ['writing_score', 'reading_score']
            cat_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]) # this is a pipeline for numerical features which first imputes the missing values with the median value and then scales the numerical features

            logging.info("numerical pipeline created successfully")

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')), 
                ('scaler', StandardScaler(with_mean=False))
            ]) # this is a pipeline for categorical features which first imputes the missing values with the most frequent value and then one hot encodes the categorical features
            logging.info("Data transformation object created successfully")
            
            preprocessor = ColumnTransformer([("num_pipeline", num_pipeline, num_features), ("cat_pipeline", cat_pipeline, cat_features)]) # this is the preprocessor object which combines the numerical and categorical pipelines
            logging.info("ColumnTransformer created successfully. Now returning the preprocessor")

            return preprocessor
        except Exception as e:
            logging.error("Error while getting data transformer object")       
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function initiates the data transformation
        '''
        try:
            logging.info("Obtaining Test and train data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Obtaining preprocessing object")
            preprocessor = self.get_data_transformer_object()

            target = 'math_score'
            num_features = ['writing_score', 'reading_score']
            cat_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            logging.info("Creating input and target dataframes")
            input_feature_train_df = train_df.drop(target, axis=1)
            target_feature_train_df = train_df[target]
            input_feature_test_df = test_df.drop(target, axis=1)
            target_feature_test_df = test_df[target]

            logging.info("Applying preprocessor to data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            logging.info("Data transformation completed successfully")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            # the following line saves the preprocessor object in the artifacts folder
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessor)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.error("Error while initiating data transformation")
            raise CustomException(e, sys.exc_info())