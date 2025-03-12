from sklearn.metrics import cohen_kappa_score

def calculate_kappa_scores(human_scores, model_scores, shut_up=False):
    """
    Calculate Unweighted, Linear Weighted, and Quadratic Weighted Cohen's Kappa.
    
    Parameters:
    human_scores (pd.Series or np.array): The ground truth labels.
    model_scores (pd.Series or np.array): The predicted labels.
    shut_up (bool): controle wheather results shall be printed to console.
    
    Returns:
    dict: Dictionary containing unweighted, linear weighted, and quadratic weighted Kappa scores.
    """
    results = {}
    results['unweighted'] = float(cohen_kappa_score(human_scores, model_scores))
    results['quadratic'] = float(cohen_kappa_score(human_scores, model_scores, weights="quadratic"))
    results['linear'] = float(cohen_kappa_score(human_scores, model_scores, weights="linear"))

    if not shut_up:
        print("Unweighted Kappa:\t\t", round(results['unweighted'], 2))
        print("Quadratic Weighted Kappa:\t", round(results['quadratic'], 2))
        print("Linear Weighted Kappa:\t\t", round(results['linear'], 2))
        
    return results