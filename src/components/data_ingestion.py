import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    """
    Data Ingestion Configuration
    """
    train_data_path: str=os.path.join(os.getcwd(), "artifacts", "train.csv") # train data path
    test_data_path: str=os.path.join("artifacts", "test.csv") # test data path
    raw_data_path: str=os.path.join("artifacts", "raw.csv") # raw data path

class DataIngestion:
    """
    Data Ingestion Class
    """
    def __init__(self, config: DataIngestionConfig):
        """
        Constructor
        """
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> None:
        """
        Initiate the data ingestion
        """
        try:
            logging.info("Initiating data ingestion")
            logging.info(os.getcwd())
            df = pd.read_csv('notebook/data/students.csv')
            
            # df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info("raw data file read successfully")

            # create the train and test data folder if it does not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) # save the train data file in the artifacts folder as train.csv file

            logging.info("train test split initated") 
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42) # split the data into train and test data
            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True) # save the train data file in the artifacts folder as train.csv file
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True) # save the test data file in the artifacts folder as test.csv file
            logging.info("train test split completed")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path, self.ingestion_config.raw_data_path) 

        except Exception as e:
            logging.error("Error while initiating data ingestion")
            raise CustomException(e, sys.exc_info())
        

if __name__ == "__main__": 
    try:
        data_ingestion = DataIngestion(DataIngestionConfig())
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.error("Error while initiating data ingestion")
        raise CustomException(e, sys.exc_info())