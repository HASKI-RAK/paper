from pathlib import Path
import pandas as pd
import zipfile
import json
import logging
from tqdm import tqdm
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional
import re


@dataclass
class EssaySetData:
    essay_set_id: int
    prompt: str
    reading_passage: str
    rubric: Dict
    source_dependent: bool

#  https://chat.deepseek.com/a/chat/s/1ae747d4-b267-4918-8ccf-df2aa18d8136


class HewlettDataset:
    """Dataset handler for Hewlett Foundation Short Answer Scoring"""

    def __init__(self, base_path: Path = Path("data/hewlett")):
        self.base_path = base_path
        self.logger = logging.getLogger(__name__)
        self.essay_sets: Dict[int, EssaySetData] = {}

        # Setup directories
        self.raw_dir = self.base_path / "raw"
        self.processed_dir = self.base_path / "processed"
        self._create_dirs()

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def _create_dirs(self):
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)

    def download(self):
        """Download dataset using Kaggle API"""
        try:
            subprocess.run([
                "kaggle", "competitions", "download",
                "-c", "asap-sas",
                "-p", str(self.raw_dir)
            ], check=True)
            self.logger.info("Dataset downloaded successfully")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Kaggle API error: {str(e)}")
            raise

    def extract(self):
        """Extract downloaded zip file"""
        zip_path = self.raw_dir / "asap-sas.zip"
        if not zip_path.exists():
            raise FileNotFoundError("No dataset zip file found")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)
        self.logger.info("Dataset extracted successfully")

    def _load_essay_sets(self):
        """Load essay set descriptions and rubrics"""
        # Implement logic to parse Data_Set_Descriptions.zip
        # This is simplified - implement based on actual data structure
        self.essay_sets = {
            1: EssaySetData(
                essay_set_id=1,
                prompt="Explain experimental procedures...",
                reading_passage="...",
                rubric={0: "Poor", 1: "Fair", 2: "Good"},
                source_dependent=True
            ),
            # Add other essay sets
        }

    def transform(self):
        """Process raw data into structured format"""
        self._load_essay_sets()

        # Load training data
        train_df = pd.read_csv(
            self.raw_dir / "train.tsv",
            sep="\t",
            usecols=["Id", "EssaySet", "EssayText", "Score1"]
        )

        # Process essays
        processed_data = []
        for essay_set_id, group in tqdm(train_df.groupby("EssaySet")):
            if essay_set_id not in self.essay_sets:
                self.logger.warning(
                    f"No metadata for essay set {essay_set_id}")
                continue

            processed_data.append({
                "essay_set_id": essay_set_id,
                "prompt": self.essay_sets[essay_set_id].prompt,
                "reading_passage": self.essay_sets[essay_set_id].reading_passage,
                "rubric": self.essay_sets[essay_set_id].rubric,
                "essays": [{
                    "essay_id": row.Id,
                    "text": self._clean_text(row.EssayText),
                    "score": row.Score1
                } for _, row in group.iterrows()]
            })

        # Save processed data
        with open(self.processed_dir / "processed.json", "w") as f:
            json.dump(processed_data, f, indent=2)

        self.logger.info("Data transformation completed")

    def load(self) -> pd.DataFrame:
        """Load processed data into DataFrame"""
        processed_path = self.processed_dir / "processed.json"
        if not processed_path.exists():
            raise FileNotFoundError("Processed data not found")

        with open(processed_path) as f:
            data = json.load(f)

        rows = []
        for essay_set in data:
            for essay in essay_set["essays"]:
                rows.append({
                    "essay_id": essay["essay_id"],
                    "essay_set": essay_set["essay_set_id"],
                    "prompt": essay_set["prompt"],
                    "text": essay["text"],
                    "score": essay["score"],
                    "rubric": json.dumps(essay_set["rubric"]),
                    "source_dependent": essay_set.get("source_dependent", False)
                })

        return pd.DataFrame(rows)

    def _clean_text(self, text: str) -> str:
        """Clean essay text"""
        # Remove extra whitespace and special markers
        cleaned = re.sub(r'\s+', ' ', text).strip()
        # Remove non-ASCII characters if needed
        return cleaned.encode('ascii', 'ignore').decode()

    def setup(self):
        """Run full pipeline"""
        self.download()
        self.extract()
        self.transform()
        return self.load()
