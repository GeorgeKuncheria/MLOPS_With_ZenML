import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluation(ABC):
    """
    Abstract class defining the structure of a model evaluation step.
    """

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the evaluation scores for the model predictions.
        Args:
            y_true: the true labels
            y_pred: the predicted labels
        
        Returns:
            None
        """
        pass



class MSE(Evaluation):
    """
    Mean Squared Error evaluation class which implements the calculate_scores method to calculate the MSE score.
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the MSE score for the model predictions.

        """
        try:
            logging.info("Calculating MSE score.")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE score calculated successfully: {}".format(mse))
            return mse
        
        except Exception as e:
            logging.error("Error in calculating MSE score: {}".format(e))
            raise e
        
    


class R2(Evaluation):
    """
    R2 evaluation class which implements the calculate_scores method to calculate the R2 score.
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the R2 score for the model predictions.

        """
        try:
            logging.info("Calculating R2 score.")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 score calculated successfully: {}".format(r2))
            return r2
        
        except Exception as e:
            logging.error("Error in calculating R2 score: {}".format(e))
            raise e
        


class RMSE(Evaluation):
    """
    Root Mean Squared Error evaluation class which implements the calculate_scores method to calculate the RMSE score.
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the RMSE score for the model predictions.

        """
        try:
            logging.info("Calculating RMSE score.")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred,squared=False))
            logging.info("RMSE score calculated successfully: {}".format(rmse))
            return rmse
        
        except Exception as e:
            logging.error("Error in calculating RMSE score: {}".format(e))
            raise e
