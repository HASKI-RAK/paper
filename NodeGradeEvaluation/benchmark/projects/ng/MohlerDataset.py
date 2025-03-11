from core.dataset import Dataset
import requests
from tqdm import tqdm
import json
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import re
from dataclasses import dataclass


@dataclass
class QuestionData:
    """Container for validated question data."""
    question_id: str
    question_text: str
    instructor_answer: str
    original_index: int
    assignment: int
    question_number: int


class QuestionValidator:
    """Handles question validation and ID mapping."""
    
    def __init__(self, logger):
        self.logger = logger
        self.question_map: Dict[str, QuestionData] = {}
        
    def extract_id(self, text: str) -> Tuple[Optional[int], Optional[int]]:
        """Extract assignment and question number from text."""
        match = re.match(r'^(\d+)\.(\d+)', text)
        if not match:
            return None, None
        return int(match.group(1)), int(match.group(2))
        
    def add_question(self, idx: int, question: str, answer: str) -> str:
        """Add question to map and return validated key."""
        assignment, q_number = self.extract_id(question)
        if not assignment or not q_number:
            self.logger.warning(f"Invalid question format at index {idx}")
            return f"IDX_{idx}"
            
        key = f"{assignment}.{q_number}"
        self.question_map[key] = QuestionData(
            question_id=key,
            question_text=question,
            instructor_answer=answer,
            original_index=idx,
            assignment=assignment,
            question_number=q_number
        )
        return key


class MohlerDataset(Dataset):
    """Dataset for Mohler's short answer grading dataset."""

    def __init__(self, name: str):
        super().__init__(name)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.validator = QuestionValidator(self.logger)

    def download(self) -> None:
        url = "http://web.eecs.umich.edu/~mihalcea/downloads/ShortAnswerGrading_v2.0.zip"
        tmp_dir = self.base_path / "tmp"
        tmp_dir.mkdir(exist_ok=True)

        zip_path = tmp_dir / "mohler_data.zip"
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            f.write(response.content)

    def extract(self) -> None:
        tmp_dir = self.base_path / "tmp"
        processed_dir = self.base_path / "processed"
        processed_dir.mkdir(exist_ok=True)

        zip_path = tmp_dir / "mohler_data.zip"
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(processed_dir)

    def transform(self) -> None:
        """Transform dataset with strict validation."""
        self.logger.info("Starting dataset transformation")
        processed_dir = self.base_path / "processed" / "data"
        output_dir = self.base_path / "processed" / "transformed"
        output_dir.mkdir(exist_ok=True)

        # Read raw data
        with open(processed_dir / "sent" / "questions") as f:
            raw_questions = [self._clean_text(line) for line in f if line.strip()]
        with open(processed_dir / "sent" / "answers") as f:
            raw_instructor_answers = [self._clean_text(line) for line in f if line.strip()]

        # Build validated question map
        for idx, (question, answer) in enumerate(zip(raw_questions, raw_instructor_answers)):
            self.validator.add_question(idx, question, answer)

        # Process student answers
        dataset = []
        files_list = self._read_files_list(processed_dir / "docs" / "files")
        
        for file_id in tqdm(files_list, desc="Processing files"):
            if result := self._process_question(processed_dir, file_id):
                dataset.append(result)

        with open(output_dir / "dataset.json", "w") as f:
            json.dump(dataset, f, indent=2)

    def load(self) -> pd.DataFrame:
        """Load the processed dataset into a pandas DataFrame."""
        processed_dir = self.base_path / "processed" / "transformed"
        data_path = processed_dir / "dataset.json"

        if not data_path.exists():
            raise FileNotFoundError("Dataset not transformed yet. Run setup() first.")

        with open(data_path) as f:
            data = json.load(f)

        # Flatten nested structure
        rows = []
        for question in data:
            for student_answer in question["student_answers"]:
                rows.append({
                    "question_id": question["id"],
                    "assignment": question["assignment"],
                    "question_number": question["question_number"],
                    "question": question["question"],
                    "instructor_answer": question["instructor_answer"],
                    "student_answer": student_answer["answer"],
                    "score": student_answer["score"],
                    "validation": question.get("validation", {})
                })

        return pd.DataFrame(rows)

    def _read_files_list(self, files_path: Path) -> List[str]:
        """Read and filter valid files from files list."""
        with open(files_path) as f:
            return [line.strip() for line in f if not line.startswith("#") and line.strip()]

    def _process_question(self, base_dir: Path, file_id: str) -> Optional[Dict]:
        """Process single question with validation."""
        try:
            # Load student data
            student_answers = self._load_student_answers(base_dir, file_id)
            scores = self._load_scores(base_dir, file_id)
            if not student_answers or not scores:
                return None

            # Get validated question data
            question_data = self.validator.question_map.get(file_id)
            if not question_data:
                self.logger.warning(f"No validated question found for {file_id}")
                return None

            return {
                "id": file_id,
                "assignment": question_data.assignment,
                "question_number": question_data.question_number,
                "question": question_data.question_text,
                "instructor_answer": question_data.instructor_answer,
                "student_answers": [
                    {"answer": answer, "score": score}
                    for answer, score in zip(student_answers, scores)
                ],
                "validation": {
                    "original_index": question_data.original_index
                }
            }
        except Exception as e:
            self.logger.error(f"Error processing {file_id}: {str(e)}")
            return None

    def _load_student_answers(self, base_dir: Path, file_id: str) -> Optional[List[str]]:
        """Load and validate student answers."""
        try:
            with open(base_dir / "sent" / file_id) as f:
                return [self._clean_text(line) for line in f if line.strip()]
        except Exception as e:
            self.logger.error(f"Failed to load student answers: {str(e)}")
            return None

    def _load_scores(self, base_dir: Path, file_id: str) -> Optional[List[float]]:
        """Load and validate scores."""
        try:
            with open(base_dir / "scores" / file_id / "ave") as f:
                return [float(line.strip()) for line in f if line.strip()]
        except Exception as e:
            self.logger.error(f"Failed to load scores: {str(e)}")
            return None

    def _clean_text(self, text: str) -> str:
        """Clean text by removing <STOP> markers and normalizing whitespace."""
        cleaned = text.replace("<STOP>", " ").strip()
        return ' '.join(cleaned.split())
