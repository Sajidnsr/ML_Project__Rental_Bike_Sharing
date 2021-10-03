import pandas as pd
import os
from application_logger.logger import applogger
from src.config import Config
import smogn
# from src.data_pull import datasetfetcher

filename = os.path.basename(__file__)

# datasetfetcher()

day = pd.read_csv("assets/original_dataset/day.csv")
hour = pd.read_csv("assets/original_dataset/hour.csv")

day = day[['instant', 'dteday', 'season', 'yr', 'mnth', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
       'casual', 'registered', 'cnt']]

hour = hour[['instant', 'dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
       'casual', 'registered', 'cnt']]

class preprocess:
    """
    This class shall be used to preprocess the data.
    """
    def __init__(self):
        self.classname = self.__class__.__name__

        self.file_object = open("Logs/data_preprocessing.txt", 'a+')
        self.logger_object = applogger()

        self.logger_object.log(self.file_object, f'Current Script: {filename}')
        self.logger_object.log(self.file_object, f'Entered the class: {self.classname}')

        Config.PROCESSED_DATASET_PATH.mkdir(parents=True, exist_ok=True)
        self.logger_object.log(self.file_object, 'Created preprocessed_data folder in assets directory')

        self.logger_object.log(self.file_object, 'Start of data preprocessing of the day data')
        
        processed_day_data = self.over_sampling(self.onehotencode(self.typecasting(self.dropcolumns(day, Config.COLS_TO_DROP), Config.DAY_CAT_COLS), Config.DAY_CAT_COLS))
        self.logger_object.log(self.file_object, 'END of data preprocessing of the day data')

        processed_day_data.to_csv(Config.PROCESSED_DATASET_PATH.joinpath('processed_day.csv'), index=False)
        self.logger_object.log(self.file_object, 'Saved processed day data to processed_day.csv')

        self.logger_object.log(self.file_object, 'Start of data preprocessing of the hour data')
        processed_hour_data = self.onehotencode(self.typecasting(self.dropcolumns(hour, Config.COLS_TO_DROP), Config.HOUR_CAT_COLS), Config.HOUR_CAT_COLS)

        self.logger_object.log(self.file_object, 'END of data preprocessing of the hour data')
        processed_hour_data.to_csv(Config.PROCESSED_DATASET_PATH.joinpath('processed_hour.csv'), index=False)
        self.logger_object.log(self.file_object, 'Saved processed day hour to processed_hour.csv')
 
    def dropcolumns(self, df, COLS_TO_DROP):
        self.funcname = self.dropcolumns.__name__
        self.logger_object.log(self.file_object, f'Entered the function: {self.funcname} of the class: {self.classname}')
        try:
            for col in df.columns:   
                if col in COLS_TO_DROP:
                    df.drop(col, axis=1, inplace=True)

            self.logger_object.log(self.file_object, f'Dropped the columns: {COLS_TO_DROP}')
            self.logger_object.log(self.file_object, f'Exited the function: {self.funcname} of the class: {self.classname}')
            return df
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in {self.funcname} method of {self.classname} class. Exception message: {e}",
            )
            raise Exception()
    def over_sampling(self, df):
        self.funcname = self.over_sampling.__name__
        self.logger_object.log(self.file_object, f'Entered the function: {self.funcname} of the class: {self.classname}')
        try:
            df = smogn.smoter(
                    data=df,
                    y='registered',
                    k=5,
                    samp_method='extreme',
                    rel_thres=0.9,
                    rel_method='auto',
                    rel_xtrm_type='high',
                    rel_coef=0.9
                )

            self.logger_object.log(self.file_object, f'Done with oversampling')
            self.logger_object.log(self.file_object, f'Exited the function: {self.funcname} of the class: {self.classname}')
            return df
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in {self.funcname} method of {self.classname} class. Exception message: {e}",
            )
            raise Exception()

    def typecasting(self, df, CAT_COLS):
        self.funcname = self.typecasting.__name__
        self.logger_object.log(self.file_object, f'Entered the function: {self.funcname} of the class: {self.classname}')
        try:
            for col in df.columns:   
                if col in CAT_COLS:
                    df[col] = df[col].astype('object')

            self.logger_object.log(self.file_object, f'Typecasted the categorical columns: {CAT_COLS}')
            self.logger_object.log(self.file_object, f'Exited the function: {self.funcname} of the class: {self.classname}')

            return df
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in {self.funcname} method of {self.classname} class. Exception message: {e}",
            )
            raise Exception()

    def onehotencode(self, df, CAT_COLS):
        self.funcname = self.onehotencode.__name__
        self.logger_object.log(self.file_object, f'Entered the function: {self.funcname} of the class: {self.classname}')
        try:
            df = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)

            self.logger_object.log(self.file_object, f'Onehot encoded the categorical columns: {CAT_COLS}')
            self.logger_object.log(self.file_object, f'Exited the function: {self.funcname} of the class: {self.classname}')

            return df
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in {self.funcname} method of {self.classname} class. Exception message: {e}",
            )
            raise Exception()
