import pandas as pd
import os
from sklearn.model_selection import train_test_split
from src.config import Config
from application_logger.logger import applogger
from src.data_preprocess import preprocess

filename = os.path.basename(__file__)

preprocess()

day = pd.read_csv("assets/processed_data/processed_day.csv")
hour = pd.read_csv("assets/processed_data/processed_hour.csv")

TEST_SIZE = 0.3

class datasplit:
    """
    This class shall be used to split the data into training and test datasets.
    """
    def __init__(self):
        self.classname = self.__class__.__name__

        self.file_object = open("Logs/split_data.txt", 'a+')
        self.logger_object = applogger()

        self.logger_object.log(self.file_object, f'Current Script: {filename}')
        self.logger_object.log(self.file_object, f'Entered the class: {self.classname}')
        
        Config.SPLIT_DATASET_PATH.mkdir(parents=True, exist_ok=True)
        self.logger_object.log(self.file_object, 'Created split_data folder in assets directory')

        self.logger_object.log(self.file_object, 'Start of day data splitting')
        day_X_train, day_X_test, day_y_train, day_y_test = self.split(day, TEST_SIZE, Config.RANDOM_STATE)
        self.logger_object.log(self.file_object, 'End of day data splitting')

        day_X_train.to_csv(Config.SPLIT_DATASET_PATH.joinpath('day_X_train.csv'), index=False)
        day_X_test.to_csv(Config.SPLIT_DATASET_PATH.joinpath('day_X_test.csv'), index=False)
        day_y_train.to_csv(Config.SPLIT_DATASET_PATH.joinpath('day_y_train.csv'), index=False)
        day_y_test.to_csv(Config.SPLIT_DATASET_PATH.joinpath('day_y_test.csv'), index=False)
        self.logger_object.log(self.file_object, 'Saved the splitted day datasets to the split_data folder')

        self.logger_object.log(self.file_object, 'Start of hour data splitting')
        hour_X_train, hour_X_test, hour_y_train, hour_y_test = self.split(hour, TEST_SIZE, Config.RANDOM_STATE)
        self.logger_object.log(self.file_object, 'End of hour data splitting')

        hour_X_train.to_csv(Config.SPLIT_DATASET_PATH.joinpath('hour_X_train.csv'), index=False)
        hour_X_test.to_csv(Config.SPLIT_DATASET_PATH.joinpath('hour_X_test.csv'), index=False)
        hour_y_train.to_csv(Config.SPLIT_DATASET_PATH.joinpath('hour_y_train.csv'), index=False)
        hour_y_test.to_csv(Config.SPLIT_DATASET_PATH.joinpath('hour_y_test.csv'), index=False)
        self.logger_object.log(self.file_object, 'Saved the splitted hour datasets to the split_data folder')

    def split(self, df, test_size, random_state):

        self.funcname = self.split.__name__
        self.logger_object.log(self.file_object, f'Entered the function: {self.funcname} of the class: {self.classname}')
        try:
            X = df.drop(columns=['casual', 'registered'], axis=1)
            y = df[['casual', 'registered']]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            self.logger_object.log(self.file_object, f'Exited the function: {self.funcname} of the class: {self.classname}')

            return X_train, X_test, y_train, y_test
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in {self.funcname} method of {self.classname} class. Exception message: {e}",
            )
            raise Exception()
