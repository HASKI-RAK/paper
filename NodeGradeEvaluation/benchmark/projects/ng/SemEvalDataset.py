from core.dataset import Dataset
import json
import logging
from typing import Optional
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset


class SemEvalDataset(Dataset):
    """Dataset for SemEval 2013 Task 7."""
    # https://huggingface.co/datasets/Atomi/sem_eval_2013_task_7

    def __init__(self, name: str):
        super().__init__(name)
        self.logger = logging.getLogger(__name__)
        self._cached_df: Optional[pd.DataFrame] = None
        self.dataset_name = "Atomi/sem_eval_2013_task_7"

    def download(self) -> None:
        """Download dataset using HuggingFace datasets."""
        self.logger.info("Starting dataset download")
        processed_dir = self.base_path / "processed"
        processed_dir.mkdir(exist_ok=True)
        file_path = processed_dir / "semeval_raw.json"

        try:
            dataset = load_dataset(self.dataset_name, split="test")
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

    def transform(self) -> None:
        """Transform dataset with progress tracking."""
        self.logger.info("Starting dataset transformation")
        processed_dir = self.base_path / "processed"

        try:
            with open(processed_dir / "semeval_raw.json") as f:
                data = json.load(f)

            transformed_data = []
            for item in tqdm(data, desc="Transforming entries"):
                try:
                    # Map classification to score
                    score = 5.0 if item.get(
                        "student_answer_label", "").upper() == "CORRECT" else 0.0

                    transformed_data.append({
                        "question_id": item.get("question_id", ""),
                        "corpus": item.get("corpus", ""),
                        "classification": item.get("classification_type", ""),
                        "question": item.get("question", ""),
                        "instructor_answer": item.get("reference_answer", ""),
                        "instructor_answer_quality": item.get("reference_answer_quality", ""),
                        "student_answer": item.get("student_answer", ""),
                        "score": score,
                        "metadata": {
                            "original_label": item.get("student_answer_label", ""),
                            "label_5way": item.get("label_5way", ""),
                            "test_set": item.get("test_set", "")
                        }
                    })
                except (KeyError, ValueError) as e:
                    self.logger.warning(f"Skipping invalid entry: {str(e)}")

            output_path = processed_dir / "semeval_transformed.json"
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
        data_path = processed_dir / "semeval_transformed.json"

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

    def extract(self) -> None:
        """Extract dataset - for SemEval, this is a no-op as data is already in JSON."""
        self.logger.info("Checking downloaded data")
        processed_dir = self.base_path / "processed"
        raw_file = processed_dir / "semeval_raw.json"

        if not raw_file.exists():
            raise FileNotFoundError(
                "Raw data file not found. Run download() first.")

        try:
            # Validate JSON structure
            with open(raw_file) as f:
                json.load(f)
            self.logger.info("Data validation successful")
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON data: {str(e)}")
            raise
