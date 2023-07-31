import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            # a pickle file is a binary file that contains the serialized version of the object. 
            # a serialized object is a byte stream that contains the object's data and enough information to reconstruct the object in memory

    except Exception as e:
        raise CustomException(e, sys)