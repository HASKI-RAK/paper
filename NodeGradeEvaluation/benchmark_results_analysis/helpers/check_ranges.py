from src import EvaluationDataClass
from loguru import logger
from typing import List, Union, Optional
import pandas as pd

def check_ranges(
    evaluation_data: EvaluationDataClass, 
    columns: List[str] = ["human_score", "model_score"], 
    output_mode: str = "print",
    include_aliases: Optional[List[str]] = None,
    exclude_aliases: Optional[List[str]] = None,
) -> Union[None, pd.DataFrame]:
    """
    Checks the minimum and maximum values of specified columns for each dataset within the EvaluationDataClass 
    and either prints this information or returns it as a pandas DataFrame.
    
    Args:
        evaluation_data (EvaluationDataClass): The object managing multiple dataset instances.
        columns (List[str], optional): List of column names to check. Defaults to ["human_score", "model_score"].
        output_mode (str, optional): Determines the output format. 
                                     Use "print" to display results or "return_df" to return a DataFrame. 
                                     Defaults to "print".
    
    Returns:
        Union[None, pd.DataFrame]: None if output_mode is "print", otherwise a DataFrame with the results.
    """
    # check if certain alias are excluded or included
    all_aliases = list(evaluation_data.datasets_metadata.keys())
    if include_aliases is not None:
        aliases_to_use = [alias for alias in all_aliases if alias in include_aliases]
    else:
        aliases_to_use = all_aliases
    if exclude_aliases is not None:
        aliases_to_use = [alias for alias in aliases_to_use if alias not in exclude_aliases]

    
    # Liste zur Speicherung der Ergebnisse
    results = []
    
    for alias, dataset in evaluation_data.datasets.items():
        # skip if shall be ignored
        if alias not in aliases_to_use:
            continue
            
        detailed_name = evaluation_data.datasets_metadata.get(alias, {}).get('detailed_name', alias)
        
        for column in columns:
            try:
                column_min = float(dataset.df[column].min())
                column_max = float(dataset.df[column].max())
                
                # Speichere die Ergebnisse in der Liste
                results.append({
                    "Dataset Alias": alias,
                    "Detailed Name": detailed_name,
                    "Column": column,
                    "Min": column_min,
                    "Max": column_max
                })
                
                if output_mode == "print":
                    print(f"Dataset: {alias} ({detailed_name})\n")
                    print(f"{column.capitalize()}\nMin: {column_min}\nMax: {column_max}\n")
                    
            except KeyError:
                logger.error(f"Column '{column}' not found in dataset '{alias}'.")
                if output_mode == "print":
                    print(f"Error: Column '{column}' not found in dataset '{alias}'.\n")
            except Exception as e:
                logger.error(f"An unexpected error occurred while processing column '{column}' in dataset '{alias}': {e}")
                if output_mode == "print":
                    print(f"An unexpected error occurred: {e}\n")
        
        if output_mode == "print":
            print("------------------------------------------------\n")
    
    if output_mode == "return_df":
        # Erstelle einen DataFrame aus den Ergebnissen
        results_df = pd.DataFrame(results)
        return results_df
    elif output_mode == "print":
        return None
    else:
        logger.error(f"Invalid output_mode '{output_mode}' specified. Use 'print' or 'return_df'.")
        raise ValueError(f"Invalid output_mode '{output_mode}' specified. Use 'print' or 'return_df'.")
