import pandas as pd
import logging
from typing import Optional

def load_csv_data(filepath: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.
    
    Args:
        filepath (str): Path to the CSV file.
        nrows (int, optional): Number of rows to read. Reads all rows if None.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(filepath, nrows=nrows)
        logging.info(f"Data loaded successfully from {filepath}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        raise

def show_basic_info(df: pd.DataFrame, n: int = 3):
    """
    Displays basic information about the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to inspect.
        n (int): Number of rows to display.
    """
    print(f"Data shape: {df.shape}")
    print(f"\nFirst {n} rows:")
    print(df.head(n))
    print("\nData types:")
    print(df.dtypes)
