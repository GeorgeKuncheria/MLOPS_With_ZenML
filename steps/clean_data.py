import logging
import pandas as pd
from zenml import step

from src.data_cleaning import DataPreprocessStrategy, DataDivideStrategy,DataCleaning
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Cleans the data by preprocessing and dividing it into train and test data.

    Args:
        df: the ingested data
    Returns:
        X_train: the training data features
        X_test: the test data features
        y_train: the training data labels
        y_test: the test data labels

    """


    try:
        process_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning step completed successfully.")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error("Error in data cleaning step: {}".format(e))
        raise e
    


