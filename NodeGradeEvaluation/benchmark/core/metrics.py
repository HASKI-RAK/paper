from typing import Any, Dict, Protocol


class MetricFunction(Protocol):
    """Protocol for metric calculation functions"""

    def __call__(self, predicted: Any, actual: Any) -> float: ...


class L1Metric:
    """Calculates the L1 metric"""

    def __call__(self, predicted: Any, actual: Any) -> float:
        return sum(abs(p - a) for p, a in zip(predicted, actual))


class AccuracyMetric:
    """Calculates the accuracy metric"""

    def __call__(self, predicted: Any, actual: Any) -> float:
        return sum(p == a for p, a in zip(predicted, actual)) / len(actual)


class ASAGMetric(MetricFunction):
    """Metric for comparing ASAG experiment results."""
    # TODO: this is bugged and does not compare old and new run

    def __call__(self, predicted: Dict, actual: Dict) -> float:
        """Calculate difference between two ASAG results."""
        # Extract scores from results structure
        pred_scores = []
        true_scores = []

        # Handle the nested structure [{'results': [{...}, {...}]}]
        if isinstance(predicted, list) and isinstance(actual, list):
            # Extract from predicted results
            for batch in predicted:
                for question in batch['results']:
                    pred_scores.extend(question['predicted_scores'])

            # Extract from actual results
            for batch in actual:
                for question in batch['results']:
                    true_scores.extend(question['true_scores'])
        else:
            raise ValueError("Invalid result structure for ASAG metric")

        if not pred_scores or not true_scores:
            raise ValueError("No scores found in results")

        # Calculate mean absolute error
        return sum(abs(p - t) for p, t in zip(pred_scores, true_scores)) / len(true_scores)
