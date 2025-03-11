from core.dataset import Dataset
import json
import logging
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from datasets import load_dataset


class EngSAFDataset(Dataset):
    """Dataset for EngSAF Short Answer Feedback.
        Transformed looks like this:
        ```json
        {
            "question_id": "1",
            "question": "What is the capital of France?",
            "instructor_answer": "Paris",
            "student_answer": "Paris",
            "score": 0
        }
        ```
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.logger = logging.getLogger(__name__)
        self._cached_df: Optional[pd.DataFrame] = None
        self.dataset_name = "dishank002/EngSAF"

    def download(self) -> None:
        """Download dataset using HuggingFace datasets."""
        self.logger.info("Starting dataset download")
        processed_dir = self.base_path / "processed"
        processed_dir.mkdir(exist_ok=True)
        file_path = processed_dir / "engsaf_raw.json"

        try:
            # only train split available
            dataset = load_dataset(self.dataset_name, split="train")
            # Convert to list for JSON serialization
            data = [dict(item)
                    for item in tqdm(dataset, desc="Processing dataset")]

            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

            self.logger.info(f"Download complete: {file_path}")
        except Exception as e:
            self.logger.error(f"Download failed: {str(e)}")
            if file_path.exists():
                file_path.unlink()
            raise

    def extract(self) -> None:
        """Validate downloaded JSON data."""
        self.logger.info("Checking downloaded data")
        processed_dir = self.base_path / "processed"
        raw_file = processed_dir / "engsaf_raw.json"

        if not raw_file.exists():
            raise FileNotFoundError(
                "Raw data file not found. Run download() first.")

        try:
            with open(raw_file) as f:
                json.load(f)
            self.logger.info("Data validation successful")
        except Exception as e:
            self.logger.error(f"Invalid JSON data: {str(e)}")
            raise

    def transform(self) -> None:
        """Transform dataset into common format."""
        self.logger.info("Starting dataset transformation")
        processed_dir = self.base_path / "processed"

        try:
            with open(processed_dir / "engsaf_raw.json") as f:
                data = json.load(f)

            transformed_data = []
            for row in tqdm(data, desc="Transforming entries"):
                try:
                    score = float(row["output_label"])

                    transformed_data.append({
                        "question_id": str(row["qid"]),
                        "question": str(row["Question"]),
                        "instructor_answer": str(row["Correct Answer"]),
                        "student_answer": str(row["Student Answer"]),
                        "score": score
                    })
                except (KeyError, ValueError) as e:
                    self.logger.warning(f"Skipping invalid entry: {str(e)}")

            # Save transformed data
            output_path = processed_dir / "engsaf_transformed.json"
            with open(output_path, "w") as f:
                json.dump(transformed_data, f, indent=2)

            self.logger.info(f"Transformed {len(transformed_data)} entries")

        except Exception as e:
            self.logger.error(f"Transformation failed: {str(e)}")
            raise

    def load(self) -> pd.DataFrame:
        """Load dataset with caching."""
        if self._cached_df is not None:
            self.logger.debug("Returning cached DataFrame")
            return self._cached_df

        processed_dir = self.base_path / "processed"
        data_path = processed_dir / "engsaf_transformed.json"

        if not data_path.exists():
            self.logger.error("Transformed dataset not found")
            raise FileNotFoundError(
                "Dataset not transformed yet. Run setup() first.")

        try:
            with open(data_path) as f:
                data = json.load(f)

            self._cached_df = pd.DataFrame(data)
            self.logger.info(f"Loaded {len(self._cached_df)} entries")
            return self._cached_df

        except Exception as e:
            self.logger.error(f"Failed to load dataset: {str(e)}")
            raise
