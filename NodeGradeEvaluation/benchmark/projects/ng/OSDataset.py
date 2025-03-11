from core.dataset import Dataset
from pathlib import Path
import json
import logging
import subprocess
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class OSQuestionData:
    """Container for validated OS dataset question data."""
    question_id: int
    answer: str
    subquestion_id: int
    question_text: str
    sample_answer: str
    sample_criteria: str
    full_points: int
    scores: List[float]
    tutorial_code_path: Optional[str] = None

    def validate(self, logger: logging.Logger) -> None:
        """Perform basic validation and log warnings if any field is missing or invalid."""
        if not self.question_text:
            logger.warning(
                f"Validation issue: question_text is empty (question_id={self.question_id})")
        if self.full_points <= 0:
            logger.warning(
                f"Validation issue: full_points is non-positive (question_id={self.question_id})")
        if not self.scores:
            logger.warning(
                f"Validation issue: no scores found (question_id={self.question_id})")


class OSDataset(Dataset):
    """Dataset handler for OS dataset with standardized processing pipeline."""

    def __init__(self, name: str):
        super().__init__(name)
        self.logger = logging.getLogger(__name__)
        self._cached_df: Optional[pd.DataFrame] = None
        self.questions: Dict[str, OSQuestionData] = {}
        self.dataset_path: Optional[Path] = None
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def download(self) -> None:
        """Download data from GitHub repository."""
        self.logger.info("Starting OS dataset download")
        repo_url = "https://github.com/wenjing1170/llm_grader.git"
        raw_dir = self.base_path / "raw"

        try:
            if raw_dir.exists():
                self.logger.info("Using existing downloaded data")
                return

            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(raw_dir)],
                check=True,
                capture_output=True
            )
            self.logger.info("Successfully downloaded OS dataset")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to download dataset: {
                              e.stderr.decode()}")
            raise

    def extract(self) -> None:
        """Extract and validate downloaded data structure."""
        self.logger.info("Starting dataset extraction")

        # Check raw directory
        raw_dir = self.base_path / "raw"
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

        # Find dataset_os directory recursively
        self.logger.info("Searching for dataset_os directory...")
        dataset_path = None
        for path in raw_dir.rglob("dataset_os"):
            if path.is_dir():
                dataset_path = path
                break

        if not dataset_path:
            raise FileNotFoundError(
                "dataset_os directory not found in repository")

        self.logger.info(f"Found dataset at: {dataset_path}")

        # Validate required subdirectories
        required_dirs = ["tutorialCriteria"]
        for dir_name in required_dirs:
            dir_path = dataset_path / dir_name
            if not dir_path.exists():
                raise FileNotFoundError(
                    f"Required directory missing: {dir_path}")

        # Store path for later use
        self.dataset_path = dataset_path
        self.logger.info("Dataset structure validation successful")

    def transform(self) -> None:
        """Transform raw data into standardized format."""
        self.logger.info("Starting dataset transformation")
        self._load_questions()

        processed_dir = self.base_path / "processed"
        processed_dir.mkdir(exist_ok=True)

        transformed_data = []
        for q_id, q_data in tqdm(self.questions.items(), desc="Transforming questions"):
            # Validate each question before output
            q_data.validate(self.logger)

            for score in q_data.scores:
                transformed_data.append({
                    "question_id": q_id,
                    "question": q_data.question_text,
                    "instructor_answer": q_data.sample_answer,
                    "student_answer": q_data.answer,
                    "score": score,
                    "metadata": {
                        "sample_criteria": q_data.sample_criteria,
                        "full_points": q_data.full_points,
                        "tutorial_code_path": q_data.tutorial_code_path,
                        "subquestion_id": q_data.subquestion_id
                    }
                })

        output_path = processed_dir / "processed.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transformed_data, f, indent=2)

        self.logger.info(f"Successfully transformed {
                         len(transformed_data)} entries")

    def load(self) -> pd.DataFrame:
        """Load processed dataset with caching."""
        if self._cached_df is not None:
            self.logger.debug("Returning cached DataFrame")
            return self._cached_df

        processed_path = self.base_path / "processed/processed.json"
        if not processed_path.exists():
            raise FileNotFoundError(
                "Dataset not transformed yet. Run setup() first.")

        try:
            with open(processed_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._cached_df = pd.DataFrame(data)
            self.logger.info(f"Loaded {len(self._cached_df)} entries")
            return self._cached_df

        except Exception as e:
            self.logger.error(f"Failed to load dataset: {str(e)}")
            raise

    def _normalize_question_text(self, question_data: dict, q_id: str) -> str:
        """Normalize question text from different formats."""
        question = question_data.get("question", "")

        if isinstance(question, list):
            self.logger.debug(
                f"Question {q_id} is in array format, joining...")
            return " ".join(question)
        elif isinstance(question, str):
            return question
        else:
            self.logger.warning(
                f"Question {q_id} has unexpected format: {type(question)}")
            return str(question)

    def _load_questions(self) -> None:
        """Load and validate questions with detailed logging."""
        if not self.dataset_path:
            raise RuntimeError("Dataset path not set. Run extract() first.")

        self.logger.info(f"Loading question data from {self.dataset_path}")

        q_dirs = [d for d in self.dataset_path.iterdir()
                  if d.is_dir() and d.name.startswith("q")]

        if not q_dirs:
            raise FileNotFoundError(
                f"No question directories found in {self.dataset_path}")

        self.logger.info(f"Found {len(q_dirs)} question directories")

        for q_dir in tqdm(q_dirs, desc="Processing questions"):
            try:
                main_id = int(q_dir.name[1:])

                # Load grading data
                grading_path = q_dir / "grading.json"
                if not grading_path.exists():
                    self.logger.warning(
                        f"Missing grading.json for {q_dir.name}")
                    continue

                with open(grading_path, "r", encoding="utf-8") as f:
                    grading_data = json.load(f)

                # Get tutorial code path if exists
                tutorial_code = self.dataset_path / "tutorialCode" / q_dir.name
                tutorial_code_path = str(
                    tutorial_code) if tutorial_code.exists() else None

                # Process nested structure
                for outer_id, outer_data in grading_data.items():
                    for inner_id, question_data in outer_data.items():
                        combined_id = f"{main_id}.{outer_id}.{inner_id}"

                        # Normalize question text
                        question_text = self._normalize_question_text(
                            question_data, combined_id)

                        scores = [
                            question_data.get(f"score_{i}", 0.0)
                            for i in range(1, 4)
                        ]

                        self.questions[combined_id] = OSQuestionData(
                            question_id=main_id,
                            subquestion_id=int(inner_id),
                            question_text=question_text,  # Use normalized text
                            answer=question_data.get("answer", ""),
                            sample_answer=question_data.get(
                                "sample_answer", ""),
                            sample_criteria=question_data.get(
                                "sample_criteria", ""),
                            full_points=question_data.get("full_points", 0),
                            scores=scores,
                            tutorial_code_path=tutorial_code_path
                        )

            except Exception as e:
                self.logger.error(f"Error processing question {
                                  q_dir.name}: {str(e)}")
                continue

        self.logger.info(f"Successfully loaded {
                         len(self.questions)} questions")
