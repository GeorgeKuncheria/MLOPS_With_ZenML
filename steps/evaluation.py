import logging
from zenml import step
import pandas as pd
from src.evaluation import MSE, R2 , RMSE

from typing import Tuple
from typing_extensions import Annotated
from sklearn.base import RegressorMixin

@step
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "mse_score"]
]:

    """
    Evaluates the trained model on the test data.

    Args:
        df: the ingested data

    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)


        return r2,mse

    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise e
