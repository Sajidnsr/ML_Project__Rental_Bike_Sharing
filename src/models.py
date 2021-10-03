import json
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from src.config import Config
from src.model_evaluation import ModelScorer
from application_logger.logger import applogger
from src.model_training import ModelTrainer
from src.data_split import datasplit
import os

filename = os.path.basename(__file__)

datasplit()

day_X_train = pd.read_csv('assets/split_data/day_X_train.csv')
day_X_test = pd.read_csv("assets/split_data/day_X_test.csv")
day_y_train = pd.read_csv("assets/split_data/day_y_train.csv")
day_y_test = pd.read_csv("assets/split_data/day_y_test.csv")
day = [day_X_train, day_X_test, day_y_train, day_y_test]

hour_X_train = pd.read_csv('assets/split_data/hour_X_train.csv')
hour_X_test = pd.read_csv("assets/split_data/hour_X_test.csv")
hour_y_train = pd.read_csv("assets/split_data/hour_y_train.csv")
hour_y_test = pd.read_csv("assets/split_data/hour_y_test.csv")
hour = [hour_X_train, hour_X_test, hour_y_train, hour_y_test]


class ModelMetricsGenerator:
    """
    This class shall be used to log the metrics for different models.
    """
    def __init__(self):
        self.classname = self.__class__.__name__
        self.file_object = open('Logs/training_model_metrics.txt', 'a+')
        self.logger_object = applogger()

        self.logger_object.log(self.file_object, f'Current Script: {filename}')
        self.logger_object.log(self.file_object, f'Entered the class: {self.classname}')

        self.models_dict = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regression": DecisionTreeRegressor(),
            "Random Forest Regression": RandomForestRegressor(),
            "Extra Trees Regression": ExtraTreesRegressor(),
        }


        day_metrics = self.create_model_metrics(day)
        self.logger_object.log(self.file_object, "Start of model training for day data.")
        with (open(str(Config.DAY_METRICS_FILE_PATH), "w")) as outfile:
                    json.dump(day_metrics, outfile, indent=1)
                
        self.logger_object.log(self.file_object, "Successfully dumped the model metrics to day_metrics.json.")

        hour_metrics = self.create_model_metrics(hour)
        self.logger_object.log(self.file_object, "Start of model training for hour data.")
        with (open(str(Config.HOUR_METRICS_FILE_PATH), "w")) as outfile:
                    json.dump(hour_metrics, outfile, indent=1)

        self.logger_object.log(self.file_object, "Successfully dumped the hour metrics to hour_metrics.json.")

    def create_model_metrics(self, lst):
        """
        This method generates a json file containing name and scores of each model.
        :return: None
        """
        self.funcname = self.create_model_metrics.__name__
        self.logger_object.log(self.file_object, f'Entered the function: {self.funcname} of the class: {self.classname}')
        try:
            self.metrics = {"models": []}
            for model_name, model in self.models_dict.items():
                model = ModelTrainer().get_trained_model(model=model, X_train=lst[0], y_train=lst[2])
                self.logger_object.log(self.file_object, f"Successfully trained {model} model.")
                r_squared, rmse = ModelScorer().get_model_scores(model=model, X_test=lst[1], y_test=lst[3])
                self.logger_object.log(self.file_object, f"Successfully calculated the r_squared and rmse for {model} model.")

                self.metrics["models"].append(
                    {"model_name": model_name, "r_squared": r_squared, "rmse": rmse}
                )
            
            self.logger_object.log(
                self.file_object,
                "Successfully appended model name and model metrics as a dictionary.",
            )
            return self.metrics['models']
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in {self.funcname} method of {self.classname} class. Exception message: {e}",
            )
            raise Exception()
        
ModelMetricsGenerator()