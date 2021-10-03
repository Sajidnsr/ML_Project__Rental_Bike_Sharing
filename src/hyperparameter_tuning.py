import numpy as np
import pandas as pd
from sklearn.ensemble import  RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from application_logger.logger import applogger
from src.config import Config
import pickle
import os

filename = os.path.basename(__file__)

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

day_parameters={'n_estimators':[int(x) for x in np.linspace(50,2000,200)],
            'max_depth':[int(x) for x in np.linspace(1,50,30)],
            'criterion':["mse", "mae"],
            'min_samples_split': [int(x) for x in np.linspace(2,50,30)],
            'min_samples_leaf': [int(x) for x in np.linspace(2,50,30)]}

hour_parameters={'n_estimators':[int(x) for x in np.linspace(50,2000,100)],
            'max_depth':[int(x) for x in np.linspace(1,50,30)],
            'criterion':["mse"],
            'min_samples_split': [int(x) for x in np.linspace(2,30,15)],
            'min_samples_leaf': [int(x) for x in np.linspace(2,30,15)]}

class hyperparametertuning:
    """
    This class shall be used for hyperparameter tuning.
    """
    def __init__(self):
        self.classname = self.__class__.__name__
        self.file_object = open("Logs/saved_models.txt", 'a+')
        self.logger_object = applogger()

        self.logger_object.log(self.file_object, f'Current Script: {filename}')
        self.logger_object.log(self.file_object, f'Entered the class: {self.classname}')

        Config.PICKLE_FILES_PATH.mkdir(parents=True, exist_ok=True)
        self.logger_object.log(self.file_object, 'Created saved_models folder in assets directory')

        day_pickle = self.tune(day, day_parameters)
        pickle.dump(day_pickle, open(Config.PICKLE_FILES_PATH.joinpath('day.sav'), 'wb'))
        self.logger_object.log(self.file_object, "Saved best estimator for day data to day.sav")

        hour_pickle = self.tune(hour, hour_parameters)
        pickle.dump(hour_pickle, open(Config.PICKLE_FILES_PATH.joinpath('hour.sav'), 'wb'))
        self.logger_object.log(self.file_object, "Saved best estimator for hour data to hour.sav")

    def tune(self, lst, params):
        self.funcname = self.tune.__name__
        self.logger_object.log(self.file_object, f'Entered the function: {self.funcname} of the class: {self.classname}')
        try:
            model_rf = RandomForestRegressor(random_state=Config.RANDOM_STATE)
            self.logger_object.log(self.file_object, "Start of hyperparameter tuning using RandomizedSearchCV")
            rfm = RandomizedSearchCV(estimator=model_rf, param_distributions=params, cv=5, n_iter=30, n_jobs=-1, verbose=5, random_state=2)
            self.logger_object.log(self.file_object, "End of hyperparameter tuning")
            rfm.fit(lst[0], lst[2])
            # print(rfm.best_score_)
            rfmod = rfm.best_estimator_

            rfmod.fit(lst[0], lst[2])

            ypred=rfmod.predict(lst[0])
            # print(r2_score(lst[2], ypred))

            ypred1=rfmod.predict(lst[1])
            # print(r2_score(lst[3], ypred1))
            self.logger_object.log(self.file_object, f'Exited the function: {self.funcname} of the class: {self.classname}')
            return rfmod
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in {self.funcname} method of {self.classname} class. Exception message: {e}",
            )
            raise Exception()
hyperparametertuning()