import logging
import pandas as pd
import numpy as np
from zenml import step


class IngestData:
    """
    Class to handle data ingestion from a CSV file."""

    def __init__(self,data_path: str):
        """Initialize the IngestData class."""
        self.data_path = data_path
    
    def get_data(self):
        """ Read data from the specified CSV file and return it as a pandas DataFrame."""
        
        
        logging.info(f'Reading data from {self.data_path}')
        data = pd.read_csv(self.data_path)
        return data



@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Step to ingest data from a CSV file.

    Args:
        data_path (str): The path to the CSV file containing the data.
    
    Returns:
        pd.DataFrame: The ingested data as a pandas DataFrame.
    """
    try:
        ingest_data=IngestData(data_path)
        df = ingest_data.get_data()
        return df
    
    except Exception as e:
        logging.error(f"Error ingesting data: {e}")
        raise e