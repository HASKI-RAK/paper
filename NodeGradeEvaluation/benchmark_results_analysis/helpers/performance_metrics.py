from src import EvaluationDataClass
from loguru import logger
from typing import List, Optional, Dict
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

def performance_metrics(
    evaluation_data: EvaluationDataClass,
    y_col: str = "human_score",
    y_hat_col: str = "model_score",
    metrics: Optional[List[str]] = None,
    include_aliases: Optional[List[str]] = None,
    exclude_aliases: Optional[List[str]] = None,
    filters: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Computes specified performance metrics for each dataset within a EvaluationDataClass.

    The function calculates metrics such as RMSE, MAE, and R2 between the actual values (Y) and
    predicted values (Y_hat) for each dataset. The columns representing Y and Y_hat as well as the metrics
    can be specified.

    Args:
        evaluation_data (EvaluationDataClass): The object managing multiple dataset instances.
        y_col (str, optional): The column name for the actual values. Defaults to "human_score".
        y_hat_col (str, optional): The column name for the predicted values. Defaults to "model_score".
        metrics (List[str], optional): List of metrics to compute. 
                                       Supported metrics: "RMSE", "MAE", "R2".
                                       Defaults to ["RMSE", "MAE", "R2"].
        include_aliases (list, optional): Only plot these dataset aliases. If None, use all unless excluded.
        exclude_aliases (list, optional): Exclude these dataset aliases. If None, exclude none.
        filters (dict, optional): A dictionary mapping dataset aliases to filter conditions. 
                                  Each filter condition should be a string formatted as a 
                                  pandas query expression, applied to the corresponding dataset. 
                                  If provided, only the data satisfying the filter will be 
                                  included in the plots. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the computed metrics for each dataset.
                      
    """
    if metrics is None:
        metrics = ["RMSE", "MAE", "R2"]

    # validate requested metrics
    supported_metrics = {"RMSE", "MAE", "R2", "Pearson"}
    invalid_metrics = set(metrics) - supported_metrics
    if invalid_metrics:
        logger.error(f"Unsupported metrics requested: {invalid_metrics}. Supported metrics are: {supported_metrics}")
        raise ValueError(f"Unsupported metrics requested: {invalid_metrics}. Supported metrics are: {supported_metrics}")

    # check if certain alias are excluded or included
    all_aliases = list(evaluation_data.datasets_metadata.keys())
    if include_aliases is not None:
        aliases_to_use = [alias for alias in all_aliases if alias in include_aliases]
    else:
        aliases_to_use = all_aliases
    if exclude_aliases is not None:
        aliases_to_use = [alias for alias in aliases_to_use if alias not in exclude_aliases]


    results = []
    for alias in aliases_to_use:
        dataset = evaluation_data.get_dataset(alias)
        df = dataset.df.copy()
        
        # check for applied filter
        if filters:
            if filters.get(alias, None):
                df = df.query(filters[alias])
                filter_applied = True # useful for later operations
                logger.info(f"Applied filter on dataset {alias}")
        else:
            filter_applied = False

        # Extract detailed name
        detailed_name = evaluation_data.datasets_metadata.get(alias, {}).get('detailed_name', alias)

        # validate if columns are present in data
        if y_col not in df.columns:
            logger.error(f"Actual value column '{y_col}' not found in dataset '{alias}'. Skipping this dataset.")
            continue
        if y_hat_col not in df.columns:
            logger.error(f"Predicted value column '{y_hat_col}' not found in dataset '{alias}'. Skipping this dataset.")
            continue

        # extract y and y_hat
        y = df[y_col]
        y_hat = df[y_hat_col]

        # calculate metrics
        metrics_result = {
            "alias": alias,
            "Detailed Name": detailed_name
        }

        try:
            if "RMSE" in metrics:
                rmse = mean_squared_error(y, y_hat)#, squared=False)
                metrics_result["RMSE"] = rmse

            if "MAE" in metrics:
                mae = mean_absolute_error(y, y_hat)
                metrics_result["MAE"] = mae

            if "R2" in metrics:
                r2 = r2_score(y, y_hat)
                metrics_result["R2"] = r2
                
            if "Pearson" in metrics:
                pearson_corr, _ = pearsonr(y, y_hat)
                metrics_result["Pearson"] = pearson_corr

            if filter_applied:
                metrics_result['filter'] = filters[alias]

        except Exception as e:
            logger.error(f"An error occurred while computing metrics for dataset '{alias}': {e}")
            continue

        results.append(metrics_result)

    results_df = pd.DataFrame(results)

    return results_df
