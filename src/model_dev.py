import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression



class Model(ABC):
    """
    Abstract class defining the structure of a machine learning model.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """Trains the model on the training data.

        Args:
            X_train: the training data features
            y_train: the training data labels
        """
        pass


class LinearRegressionModel(Model):
    """
    Linear regression model which implements the train method to train a linear regression model.
    """

    
    
    def train(self, X_train, y_train,**kwargs):
        """Trains the linear regression model on the training data.

        Args:
            X_train: the training data features
            y_train: the training data labels
        """
        try:
            self.reg = LinearRegression(**kwargs)
            self.reg.fit(X_train, y_train)
            logging.info("Model training step completed successfully.")
            return self.reg
        
        except Exception as e:
            logging.error("Error in model training step: {}".format(e))
            raise e
        
