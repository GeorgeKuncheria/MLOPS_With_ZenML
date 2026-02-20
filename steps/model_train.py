import logging
from zenml import step
import pandas as pd
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
# from .config import ModelNameConfig


@step
def train_model(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame,
    config: str
) -> RegressorMixin:
    """
    Trains the model on the ingested data
    
    Args:
        df: the ingested data as a pandas DataFrame
    
    """

    try:
        if X_train.isnull().values.any():
            logging.warning("X_train still contains NaNs! Cleaning might have failed.")
            X_train = X_train.fillna(0) # Emergency fallback
        
        model = None
        if config == "LinearRegression":
            model = LinearRegressionModel()
            trained_model=model.train(X_train, y_train)
            logging.info("Model training step completed successfully.")
            return trained_model
        

        else:
            raise ValueError("Model {} not supported.".format(config))
    
    except Exception as e:
        logging.error("Error in model training step: {}".format(e))
        raise e
    