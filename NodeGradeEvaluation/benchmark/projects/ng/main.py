"""
This module contains the NodeGradeEvaluation project that evaluates student answers using the benchmark API.
    The experiments can be parameterized. Results are saved in a "run_history" folder.
    If you try to launch this project, start from the root folder of the project, not inside the projects or ng folder.
Take a look at the README.md on how to run the project.
"""
import json
from pathlib import Path
import requests
from typing import Any, Dict, List
from tqdm import tqdm
import pandas as pd

from core.decorators import experiment
from core.experiment_config import ExperimentConfig
from core.project import Project
import time
import numpy as np
from sklearn.metrics import cohen_kappa_score

from projects.ng.SemEvalDataset import SemEvalDataset
from projects.ng.MohlerDataset import MohlerDataset
from projects.ng.EngSAFDataset import EngSAFDataset
from projects.ng.OSDataset import OSDataset

# constants
# Enter the url of the NodeGrade tool here. Information on how to set it up can be
# found at the repository: https://github.com/HASKI-RAK/NodeGrade
API_URL = "http://193.174.195.36:5005/v1/benchmark"

class NodeGradeEvaluation(Project):
    """Main project. Here datasets are registered and evaluation per function is performed.
    """

    def __init__(self, name: str, project_root: Path):
        super().__init__(name, project_root)

        self.dataset_manager.register(MohlerDataset("mohler"))
        self.dataset_manager.register(SemEvalDataset("semeval"))
        self.dataset_manager.register(EngSAFDataset("engsaf"))
        self.dataset_manager.register(
            OSDataset("os_dataset"))
        self.api_url = API_URL

    def _call_grading_api(self, question: str, real_answer: str, student_answer: str, path="/ws/editor/strategie_leicht_benchmark/1/1") -> float:
        """Make REST call to grading endpoint with retry logic.
        Returns raw score (0-100)."""
        payload = {
            "path": path,
            "data": {
                "question": question,
                "realAnswer": real_answer,
                "answer": student_answer
            }
        }
        max_retries = 3
        score = 0.0  # Default score

        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, json=payload)
                response.raise_for_status()
                data = response.json()
                # Handle both list and dict response formats
                if isinstance(data, list):
                    return float(data[0]) if data else 0.0
                return float(data.get('score', 0.0))
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"API call failed after {max_retries} attempts: {e}")
                    # details about which question failed
                    print(f"Failed question: {
                          question} - {real_answer} - {student_answer}")
                    print(f"Returned score: {score}")
                    return 0.0
                time.sleep(1)  # Wait before retry

        return score  # Return raw score

    @experiment("ng_test",
                ExperimentConfig("ng_test")
                .add_input({"data": {
                    "question": "Why is the interface not working?",
                    "realAnswer": "they are too different",
                    "answer": "they are really different"
                }}, description="Test request for the evaluation"))
    def run_testeval_ng(self, data: dict) -> dict:
        """Test request for the evaluation
        """
        return data

    @experiment("ng_mohler",
                ExperimentConfig("ng_mohler")
                # this path refers to the graph configuraiton loaded by the tool.
                .add_input(path="/ws/editor/strategie_leicht_benchmark_de/1/1",
                           description="Evaluate student answers using Mohler dataset")
                # This is just a unique identifier for the dataset
                .add_dataset("mohler", "2.0"))
    def run_mohler_dataset(self, path: str) -> List[Dict]:
        """Run evaluation on Mohler dataset."""
        dataset = self.dataset_manager.get("mohler")
        df = dataset.get_data()
        wait_sanity(df)

        results = []
        total_answers = sum(1 for _, row in df.iterrows())

        with tqdm(total=total_answers, desc="Processing Mohler answers") as pbar:
            for _, row in df.iterrows():
                model_score = self._call_grading_api(
                    question=row["question"],
                    real_answer=row["instructor_answer"],
                    student_answer=row["student_answer"],
                    path=path
                )

                results.append({
                    'question_id': row["question_id"],
                    'assignment': row["assignment"],
                    'question_number': row["question_number"],
                    'model_score': model_score,
                    'human_score': row["score"],
                    'question': row["question"],
                    'instructor_answer': row["instructor_answer"],
                    'student_answer': row["student_answer"]
                })
                pbar.update(1)

        return results

    @experiment("ng_semeval",
                ExperimentConfig("ng_semeval")
                .add_input(path="/ws/editor/strategie_leicht_benchmark/1/1",
                           description="Evaluate student answers using SemEval dataset")
                .add_dataset("semeval", "1.1"))
    def run_semeval_dataset(self, path: str) -> List[Dict]:
        """Run evaluation on SemEval dataset."""
        dataset = self.dataset_manager.get("semeval")
        df = dataset.get_data()
        # df = df.sample(10)  # Sample 5 random rows for testing
        wait_sanity(df)

        # Validate required columns
        required_columns = ["question_id", "question", "instructor_answer",
                            "student_answer", "score"]
        missing_columns = [
            col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        results = []
        with tqdm(total=len(df), desc="Processing SemEval answers") as pbar:
            for _, row in df.iterrows():
                model_score = self._call_grading_api(
                    question=row["question"],
                    real_answer=row["instructor_answer"],
                    student_answer=row["student_answer"],
                    path=path
                )

                # Build result with optional metadata
                result = {
                    'question_id': row["question_id"],
                    'model_score': model_score,
                    'human_score': row["score"],
                    'question': row["question"],
                    'instructor_answer': row["instructor_answer"],
                    'student_answer': row["student_answer"]
                }

                # Add metadata if available
                metadata = {}
                metadata_fields = {
                    'corpus': 'corpus',
                    'classification': 'classification',
                    'instructor_answer_quality': 'instructor_answer_quality',
                    'original_label': ['metadata', 'original_label'],
                    'label_5way': ['metadata', 'label_5way'],
                    'test_set': ['metadata', 'test_set']
                }

                for key, field in metadata_fields.items():
                    try:
                        if isinstance(field, list):
                            value = row[field[0]][field[1]]
                        else:
                            value = row[field]
                        metadata[key] = value
                    except (KeyError, TypeError):
                        continue

                if metadata:
                    result['metadata'] = metadata

                results.append(result)
                pbar.update(1)

        return results

    @experiment("ng_engsaf",
                ExperimentConfig("ng_engsaf")
                .add_input(path="/ws/editor/strategie_leicht_benchmark_de/1/1",
                           description="EngSAF dataset evaluation with german prompt")
                .add_input(path="/ws/editor/strategie_leicht_benchmark/1/1",
                           #    samples=5,
                           description="EngSAF dataset evaluation with english prompt")
                .add_dataset("engsaf", "1.0"))
    def run_engsaf_dataset(self, path: str) -> List[Dict]:
        """Run evaluation on EngSAF dataset."""
        dataset = self.dataset_manager.get("engsaf")
        df = dataset.get_data()
        wait_sanity(df)
        results = []
        with tqdm(total=len(df), desc="Processing EngSAF answers") as pbar:
            for _, row in df.iterrows():
                model_score = self._call_grading_api(
                    question=row["question"],
                    real_answer=row["instructor_answer"],
                    student_answer=row["student_answer"],
                    path=path
                )

                results.append({
                    'question_id': row["question_id"],
                    'model_score': model_score,
                    'human_score': row["score"],
                    'question': row["question"],
                    'instructor_answer': row["instructor_answer"],
                    'student_answer': row["student_answer"]
                })
                pbar.update(1)

        return results

    @experiment("ng_os",
                ExperimentConfig("ng_os")
                .add_input(path="/ws/editor/strategie_leicht_benchmark_de/1/1",
                           description="Evaluate student answers using OS dataset")
                .add_dataset("os_dataset", "1.0"))
    def run_os_dataset(self, path: str) -> List[Dict]:
        """Run evaluation on OS dataset."""
        dataset = self.dataset_manager.get("os_dataset")
        df = dataset.get_data()

        # Validate required columns
        required_columns = ["question", "instructor_answer",
                            "student_answer", "score", "metadata"]
        missing_columns = [
            col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        print(f"Available columns: {list(df.columns)}")
        wait_sanity(df)

        results = []
        total_answers = len(df)

        with tqdm(total=total_answers, desc="Processing OS dataset answers") as pbar:
            for _, row in df.iterrows():
                try:
                    metadata = row["metadata"]
                    model_score = self._call_grading_api(
                        question=row["question"],
                        real_answer=row["instructor_answer"],
                        student_answer=row["student_answer"],
                        path=path
                    )

                    results.append({
                        'question_id': row["question_id"],
                        'model_score': model_score,
                        'human_score': row["score"],
                        'question': row["question"],
                        'instructor_answer': row["instructor_answer"],
                        'student_answer': row["student_answer"],
                        'metadata': {
                            'full_points': metadata.get('full_points', 0),
                            'sample_criteria': metadata.get('sample_criteria', ''),
                            'tutorial_code_path': metadata.get('tutorial_code_path', None),
                            'subquestion_id': metadata.get('subquestion_id', None)
                        }
                    })
                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing row: {str(e)}")
                    continue

        print(f"Successfully processed {len(results)} answers")
        return results
    
    def evaluate(self, results: List[Dict]) -> Dict[str, float]:
        """Evaluates the results by comparing model scores to human scores.
        Normalizes API scores from 0-100 to 0-5 scale.
        To perform the results for our paper, we used additional jupyter notebooks
        which can be found in the neighboring folders.
        """
        if not results:
            return {'pearson': 0.0, 'mse': 0.0, 'weighted_kappa': 0.0}

        try:
            # Normalize model scores from 0-100 to 0-5
            model_scores = np.array(
                [result['model_score'] / 20.0 for result in results])
            human_scores = np.array([result['human_score']
                                    for result in results])

            # Calculate metrics with normalized scores
            pearson_corr = np.corrcoef(model_scores, human_scores)[0, 1]
            mse = np.mean((model_scores - human_scores) ** 2)

            # Convert normalized scores to categories
            bins = np.linspace(0, 5, num=6)
            model_scores_binned = np.digitize(model_scores, bins=bins)
            human_scores_binned = np.digitize(human_scores, bins=bins)

            weighted_kappa = cohen_kappa_score(
                human_scores_binned,
                model_scores_binned,
                weights='quadratic'
            )

            return {
                'pearson': float(pearson_corr),
                'mse': float(mse),
                'weighted_kappa': float(weighted_kappa)
            }

        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return {'pearson': 0.0, 'mse': 0.0, 'weighted_kappa': 0.0}

    def show_output(self, results: List[Dict]) -> None:
        """This function shows the evaluation results in a human-readable format.
        While it is not necessary for the evaluation, it is useful for debugging.
        """
        metrics = self.evaluate(results)
        print("\nEvaluation Results:")
        print("=" * 50)
        print(f"Pearson correlation: {metrics['pearson']:.3f}")
        print(f"Mean squared error: {metrics['mse']:.3f}")
        print(f"Weighted Kappa: {metrics['weighted_kappa']:.3f}")
        print("-" * 50)
        print(f"Processed {len(results)} answers")

        # Add per-question analysis
        questions = {}
        for r in results:
            qid = r['question_id']
            if qid not in questions:
                questions[qid] = {'model': [], 'human': []}
            questions[qid]['model'].append(r['model_score'])
            questions[qid]['human'].append(r['human_score'])

        print("\nPer-Question Analysis (First 5 Questions):")
        print("=" * 50)
        for i, (qid, scores) in enumerate(questions.items()):
            if i >= 5:
                break
            pearson = np.corrcoef(scores['model'], scores['human'])[0, 1]
            print(f"\nQuestion {qid}:")
            print(f"Pearson correlation: {pearson:.3f}")
            print(f"Number of answers: {len(scores['model'])}")

    def output_metrics_and_results(self, results: Any) -> None:
        """Output detailed metrics and results for the experiment.

        Args:
            results: Can be either List[Dict] or Tuple[List[Dict], Dict]
        """
        if isinstance(results, tuple) and len(results) == 2:
            # Extract results from (results, params) tuple
            results, _ = results

        # Handle nested list structure from JSON
        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
            results = results[0]  # Unwrap outer list

        # Type check before passing to show_output
        if not isinstance(results, list) or not all(isinstance(r, dict) for r in results):
            raise ValueError("Invalid results format")

        self.show_output(results)  # Now we know results is List[Dict]


def wait_sanity(df: pd.DataFrame, samples: int = 5) -> None:
    """Wait for user input to continue.
    This helps to interrupt the program in case of a mistake.
    """
    # Sanity check by printing the first 5 rows and wait for user input
    print(df.head(samples))
    print("Press Ctrl+C to stop or wait for 5 seconds to continue...")
    time.sleep(5)
