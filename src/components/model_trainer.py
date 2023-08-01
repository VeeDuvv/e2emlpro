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

    def initiate_model_trainer(self, train_arr, test_arr, preprocessor_path):
        try:
            logging.info("Initiating model training - splitting training data into features and target")
            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

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
            logging.info("Evaluate models function completed... the best model is...")
            best_model = model_report["model_name"][model_report["r2_score"].argmax()]

            if model_report{"r2_score"}.max() < 0.8:
                logging.info("Best model has r2 score less than 0.8, hence not saving the model")
                return
            logging.info("Saving the best model")
            save_object(file_path=self.model_trainer_config.trained_model_file_path, model=models[best_model], preprocessor_path=preprocessor_path)
            logging.info("Model saved successfully")
        except Exception as e:
            logging.error("Error while initiating model training")
            raise CustomException(e, sys.exc_info())