import tikzplotlib  
from matplotlib import pyplot as plt

def optimize_threshold(y_true_binary, y_pred_score, thresholds, optimize_for="f1"):
    """
    Optimizes the decision threshold for a given metric and calculates other metrics at this optimum.
    
    Parameters:
    -----------
    y_true_binary : array-like
        The ground truth binary labels (0 or 1).
    
    y_pred_score : array-like
        The predicted scores (probabilities or confidence scores).
    
    thresholds : list or array-like
        A list of threshold values to evaluate.
    
    optimize_for : str, optional, default="f1"
        The metric to optimize for. Must be one of ["accuracy", "recall", "precision", "f1"].
    
    Returns:
    --------
    results : dict
        A dictionary containing the optimal threshold, the maximum optimized metric value,
        and the other calculated metrics at the best threshold.
    
    metric_values : list
        A list of the metric values computed for each threshold.
    
    Raises:
    -------
    ValueError
        If `optimize_for` is not one of the allowed metrics.
    """
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
    import numpy as np
    
    metrics = {
        "accuracy": accuracy_score,
        "recall": recall_score,
        "precision": precision_score,
        "f1": f1_score
    }
    
    if optimize_for not in metrics:
        raise ValueError(f"Invalid metric '{optimize_for}'. Choose from {list(metrics.keys())}.")
    
    # 1) Metrik-Werte über alle thresholds ausrechnen
    metric_values = []
    for threshold in thresholds:
        predicted_labels = (y_pred_score >= threshold).astype(int)
        metric_value = metrics[optimize_for](y_true_binary, predicted_labels)
        metric_values.append(metric_value)

    # 2) Bestes Threshold finden
    best_index = np.argmax(metric_values)
    best_threshold = thresholds[best_index]
    best_metric_value = metric_values[best_index]

    # 3) Andere Metriken an diesem Threshold berechnen
    predicted_labels = (y_pred_score >= best_threshold).astype(int)
    results = {
        "best_threshold": best_threshold,
        f"max_{optimize_for}": best_metric_value
    }
    for metric_name, metric_func in metrics.items():
        results[metric_name] = metric_func(y_true_binary, predicted_labels)

    # Print human-readable results
    print("\nOptimal threshold analysis:")
    print(f"Optimized for: {optimize_for.upper()}")
    print(f"Best threshold found: {best_threshold:.3f}")
    print(f"Maximum {optimize_for}: {best_metric_value:.3f}")
    
    print("\nPerformance at best threshold:")
    for metric_name, metric_value in results.items():
        if metric_name != "best_threshold":
            print(f"{metric_name.capitalize()}: {metric_value:.3f}")
    
    return results, metric_values

def plot_metric_vs_threshold(thresholds, metric_values, best_threshold, metric_name="f1", file_path_prefix=None):
    """
    Plots the specified metric against the threshold values and highlights the optimal threshold.
    
    Parameters:
    -----------
    thresholds : list or array-like
        A list of threshold values used for evaluation.
    
    metric_values : list or array-like
        The corresponding metric values computed for each threshold.
    
    best_threshold : float
        The threshold that maximizes the specified metric.
    
    metric_name : str, optional, default="f1"
        The name of the metric being plotted. Used for labeling the graph.
    
    Returns:
    --------
    None
        The function generates a plot but does not return any values.
    """


    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, metric_values, label=metric_name.capitalize())
    plt.axvline(x=best_threshold, color='r', linestyle='--', 
                label=f'Best {metric_name} = {best_threshold:.3f}')
    plt.xlabel("Threshold")
    plt.ylabel(metric_name.capitalize())
    plt.title(f"{metric_name.capitalize()} vs. Threshold")
    plt.legend(loc='best')
    plt.grid()
    if file_path_prefix:
        plt.savefig(f"{file_path_prefix}.png")
        tikzplotlib.save(f"{file_path_prefix}.tex")
    plt.show()

