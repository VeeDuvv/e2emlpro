import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    '''
    Evaluate the model
    '''
    try:
        report = {}
        for i in range(len(list(models.keys()))):
            model_name = list(models.keys())[i]
            param = params[model_name]
            model = list(models.values())[i]
            # GridSearchCV is a method to find the best parameters for a model. 
            # It does this by fitting the model multiple times on the provided dataset with different parameter values. 
            # Its parameters are:   estimator: the model we want to fit, param_grid: the dictionary of parameters we want to optimize, 
            # cv: the number of folds we want to use for cross validation
            gs = GridSearchCV(model,param,cv=3) 
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report[model_name] = r2_score(y_test, y_pred)
            logging.info(f"Model: {model_name}, R2 Score: {report[model_name]}")
    except Exception as e:
        raise CustomException(e, sys)
    return report