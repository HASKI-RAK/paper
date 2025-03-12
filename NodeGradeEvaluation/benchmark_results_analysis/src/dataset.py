# ./src/dataset.py
import json
import pandas as pd
import sys
from loguru import logger
from typing import Any, Dict, List, Optional


class Dataset:
    """
    A class to encapsulate a single dataset, including metadata, parameters, and results.
    """

    def __init__(self, file_path: str, additional_metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the Dataset object by loading data from a JSON file.
    
        Args:
            file_path (str): Path to the JSON file containing evaluation data.
            additional_metadata (dict, optional): Additional metadata from external configuration.
        """
        self.file_path = file_path
        self.metadata: Dict[str, Any] = additional_metadata if additional_metadata else {}
        self.parameters: Dict[str, Any] = {}
        self.results: List[Dict[str, Any]] = []
        self.df: pd.DataFrame = pd.DataFrame()
    
        # Load data
        self._load_data()
    
        # Create dataframe
        self.df = self._create_dataframe()
        logger.debug("Dataset object initialized successfully.")

    def _load_data(self) -> None:
        """
        Load evaluation data from the JSON file specified by file_path.
        """
        try:
            logger.info(f"Loading evaluation data from {self.file_path}")
            with open(self.file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
    
            # Merge the additional metadata with the loaded metadata
            file_metadata = data.get('metadata', {})
            self.metadata = {**self.metadata, **file_metadata}
    
            self.parameters = data.get('parameters', {})
            raw_results = data.get('results', [])
    
            # Flatten the results if it's a list of lists
            if any(isinstance(item, list) for item in raw_results):
                self.results = [result for sublist in raw_results for result in sublist]
                logger.debug("Flattened the results list.")
            else:
                self.results = raw_results
    
            logger.debug(f"Metadata: {self.metadata}")
            logger.debug(f"Parameters: {self.parameters}")
            logger.debug(f"Number of results: {len(self.results)}")
    
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in file: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise
   
    def _create_dataframe(self) -> pd.DataFrame:
        """
        Convert the list of result dictionaries into a pandas DataFrame.
        If `flatten_json_in_results` is defined, flatten the specified JSON objects in the results.
        
        Returns:
            pd.DataFrame: DataFrame containing the results.
        """
        logger.debug("Creating DataFrame from results.")
        try:
            # Convert the results to a DataFrame
            df = pd.DataFrame(self.results)
    
            # Check if the dataset metadata includes fields to flatten
            flatten_fields = self.metadata.get("flatten_json_in_results", [])

            # Flatten specified fields in the results
            for field in flatten_fields:
                if field in df.columns:
                    logger.debug(f"Flattening field '{field}' in the DataFrame.")
                    
                    # Expand the JSON column into individual columns
                    flattened = pd.json_normalize(df[field])
                    
                    # Prefix the new columns with the field name to avoid conflicts
                    flattened.columns = [f"{field}_{col}" for col in flattened.columns]
                    
                    # Add the flattened columns to the DataFrame
                    df = pd.concat([df.drop(columns=[field]), flattened], axis=1)
    
            logger.debug(f"DataFrame created with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to create DataFrame: {e}")
            return pd.DataFrame()
    

    def export_to_csv(self, output_path: str) -> None:
        """
        Export the DataFrame to a CSV file.

        Args:
            output_path (str): Path where the CSV file will be saved.
        """
        try:
            self.df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Data exported to CSV at {output_path}")
        except Exception as e:
            logger.error(f"Failed to export data to CSV: {e}")
            raise

