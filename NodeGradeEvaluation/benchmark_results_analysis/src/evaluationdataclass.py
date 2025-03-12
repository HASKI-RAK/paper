from typing import Dict
from loguru import logger
from .dataset import Dataset
from typing import Any, Dict

class EvaluationDataClass:
    """
    A class to manage multiple Dataset instances along with their metadata.
    """

    def __init__(self, datasets_metadata: Dict[str, Dict[str, Any]]):
        """
        Initialize the EvaluationDataClass with multiple datasets.

        Args:
            datasets_metadata (dict): Dictionary containing metadata and file paths for each dataset.
        """
        self.datasets_metadata = datasets_metadata
        self.datasets: Dict[str, Dataset] = {}
        self._initialize_datasets()
        logger.debug("EvaluationDataClass initialized with multiple datasets.")
    
    def _initialize_datasets(self):
        """
        Initialize all Dataset instances based on the provided metadata.
        """
        for alias, metadata in self.datasets_metadata.items():
            try:
                logger.debug(f"Initializing dataset '{alias}' from {metadata['filepath']}")
                dataset = Dataset(
                    file_path=metadata['filepath'],
                    additional_metadata=metadata
                )
                self.datasets[alias] = dataset
                logger.debug(f"Dataset '{alias}' loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load dataset '{alias}': {e}")


    def get_dataset(self, alias: str) -> Dataset:
        """
        Retrieve a specific Dataset by its alias.

        Args:
            alias (str): The alias of the dataset.

        Returns:
            Dataset: The requested Dataset instance.
        """
        return self.datasets.get(alias)

    def export_all_to_csv(self, output_dir: str):
        """
        Export all datasets to CSV files in the specified directory.

        Args:
            output_dir (str): Directory where CSV files will be saved.
        """
        for alias, dataset in self.datasets.items():
            output_path = f"{output_dir}/{alias}.csv"
            dataset.export_to_csv(output_path)
            logger.info(f"Exported dataset '{alias}' to {output_path}")
