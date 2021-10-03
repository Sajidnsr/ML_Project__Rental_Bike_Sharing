from sklearn.metrics import mean_squared_error, r2_score

from application_logger.logger import applogger


class ModelScorer:
    """
    This class shall be used for getting scores of the trained model.
    """

    def __init__(self):
        self.classname = self.__class__.__name__
        self.file_object = open('Logs/training_model_metrics.txt', 'a+')
        self.logger_object = applogger()

    def get_model_scores(self, model, X_test, y_test):
        """
        This method is used for evaluating the trained model.
        :param model: trained model to be evaluated
        :param X_test: features
        :param y_test: labels
        :return: r squared score and root mean squared error
        """
        self.funcname = self.get_model_scores.__name__
        self.logger_object.log(
            self.file_object, f"Entered the function: {self.funcname} of the class: {self.classname}"
        )
        try:
            y_pred = model.predict(X_test)
            # self.logger_object.log(self.file_object, "Successfully predicted X_test.")
            r_squared = r2_score(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            # self.logger_object.log(self.file_object, "Successfully calculated model scores.")

            self.logger_object.log(
                    self.file_object, f"Exited the function: {self.funcname} of the class: {self.classname}"
                )

            return r_squared, rmse
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                f"Exception occured in {self.funcname} method of {self.classname} class. Exception message: {e}",
            )
            raise Exception()

            
