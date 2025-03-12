from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report, matthews_corrcoef
)
import numpy as np

def evaluate_classification(y_true, y_pred):
    """
    Evaluates a binary classification model using common performance metrics.
    
    Parameters:
    y_true (array-like): True labels (0 or 1)
    y_pred (array-like): Predicted labels (0 or 1)
    
    Returns:
    dict: Dictionary containing all performance metrics
    """
    # Compute classification metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True)
    }
    
    # Print metrics
    print("Classification Metrics:")
    for key, value in metrics.items():
        if key not in ["confusion_matrix", "classification_report"]:
            print(f"{key}: {value:.4f}")
    
    print("Confusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))
    
    return metrics