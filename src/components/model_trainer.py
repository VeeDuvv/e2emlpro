import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str=os.path.join("artifacts", "model.pkl") # trained model path

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Initiating model training - splitting training data into features and target")
            
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            logging.info("Calling evaluate models function")
            model_report:dict=evaluate_models(X_train, y_train, X_test, y_test, models)
            # from a dictionary, find the key corresponding to the maximum value
            best_model = max(model_report, key=model_report.get)
            best_model_r2 = model_report[best_model]
            logging.info("Evaluate models function completed...")
            logging.info(f"Best model is: {best_model} with r2 score: {best_model_r2}")
            
            if best_model_r2 < 0.8:
                logging.info("Best model has r2 score less than 0.8, hence not saving the model")
                return
            logging.info("Saving the best model")
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=models[best_model])
            logging.info("Model saved successfully")


        except Exception as e:
            logging.error("Error while initiating model training")
            raise CustomException(e, sys.exc_info())