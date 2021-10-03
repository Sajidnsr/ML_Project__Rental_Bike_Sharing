from src.config import Config
from application_logger.logger import applogger
from sklearn.model_selection import RandomizedSearchCV

class ModelTrainer:
    """
    This class shall be used for training the model.
    """

    def __init__(self):
        self.classname = self.__class__.__name__
        self.file_object = open('Logs/training_model_metrics.txt', 'a+')
        self.logger_object = applogger()

    def get_trained_model(self, model, X_train, y_train):
        """
        This method is used for training the model.
        :param model: model to be trained
        :param X_train: features
        :param y_train: labels
        :return: trained model
        """
        self.funcname = self.get_trained_model.__name__
        try:
            model = model.fit(X_train, y_train)
            return model 
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in {self.funcname} method of {self.classname} class. Exception message: {e}",
            )
            raise Exception()       