import logging
from src.data_pipeline.data_loader import load_csv_data
from src.data_pipeline.cleaner import clean_data
from src.data_pipeline.feature_engineer import engineer_features

class FeedETLPipeline:
    def __init__(self, input_path: str, output_path: str = None):
        self.input_path = input_path
        self.output_path = output_path or input_path.replace('.csv', '_etl.csv')
        self.df = None

    def run(self):
        logging.info("Starting ETL pipeline...")
        self.load_data()
        self.clean_data()
        self.feature_engineering()
        self.save_data()
        logging.info("ETL pipeline completed successfully.")

    def load_data(self):
        self.df = load_csv_data(self.input_path)
        logging.info(f"Loaded data: {self.df.shape[0]} rows, {self.df.shape[1]} columns")

    def clean_data(self):
        self.df = clean_data(self.df)
        logging.info("Data cleaning complete.")

    def feature_engineering(self):
        self.df = engineer_features(self.df)
        logging.info("Feature engineering complete.")

    def save_data(self):
        self.df.to_csv(self.output_path, index=False)
        logging.info(f"Saved cleaned and feature-rich data to {self.output_path}")
