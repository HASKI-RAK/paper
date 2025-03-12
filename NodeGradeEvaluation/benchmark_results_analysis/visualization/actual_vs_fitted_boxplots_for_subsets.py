import matplotlib.pyplot as plt
import pandas as pd
from math import ceil
from loguru import logger
from typing import Optional, Dict, List
import tikzplotlib

def actual_vs_fitted_boxplots_for_subsets(
    evaluation_data,
    alias: str,
    subset_col: str,
    layout: str = "horizontal",
    x_col: str = "human_score",
    y_col: str = "model_score",
    x_label: str = "Human Score",
    y_label: str = "Model Score",
    format_x_labels: bool = True,
    filters: Optional[str] = None,
    order_overrides: Optional[List] = None,
    output_filepath_prefix: Optional[str] = None
):
    """
    Creates multiple boxplots (one per subset in a specified column) for a single dataset 
    within an EvaluationDataClass instance. Subplots are organized in a single figure.

    Args:
        evaluation_data (EvaluationDataClass): The EvaluationDataClass object containing 
                                               multiple datasets.
        alias (str): The alias of the dataset to be plotted.
        subset_col (str): The name of the column indicating subsets (e.g., questions). 
                          A boxplot will be created for each unique value in this column.
        layout (str): Layout of the plots, either 'horizontal' or 'vertical'.
                      'horizontal' attempts to create 2 columns for subplots.
                      'vertical' uses 1 column (stacked).
        x_col (str): Column name for the grouping (X-axis). Defaults to "human_score".
        y_col (str): Column name for the predicted/target values (Y-axis). Defaults to "model_score".
        x_label (str): Label for the X-axis. Defaults to "Human Score".
        y_label (str): Label for the Y-axis. Defaults to "Model Score".
        format_x_labels (bool): If True, format numeric X-axis labels (integer vs. float). 
                                Defaults to True.
        filters (str, optional): A string formatted as a pandas query expression, applied 
                                 to the dataset before plotting. Example: "topic == 'Math'".
        order_overrides (list, optional): An explicit list defining the category order for the 
                                          x-axis (x_col). If provided, x_col is treated as 
                                          a Categorical in this order.
        output_filepath_prefix (str, optional): If provided, the entire multi-subplot figure 
                                                is saved as a PNG and a TEX (via tikzplotlib). 
                                                Defaults to None.

    Returns:
        None
    """
    # 1) Retrieve the dataset for the specified alias
    dataset = evaluation_data.get_dataset(alias)
    if dataset is None:
        logger.error(f"Dataset '{alias}' not found in EvaluationDataClass.")
        return

    data = dataset.df.copy()
    metadata = evaluation_data.datasets_metadata.get(alias, {})
    detailed_name = metadata.get('detailed_name', alias)

    # 2) Optional filtering
    if filters:
        try:
            data = data.query(filters).copy()
        except Exception as e:
            logger.error(
                f"Failed to apply filter '{filters}' to dataset '{alias}': {e}"
            )
            return

    # 3) Check presence of required columns
    for col_needed in [subset_col, x_col, y_col]:
        if col_needed not in data.columns:
            logger.error(
                f"Column '{col_needed}' not found in dataset '{alias}'. "
                f"Skipping this dataset."
            )
            return

    # 4) Identify all unique subsets
    unique_subsets = sorted(data[subset_col].unique())

    # If there are no subsets, nothing to plot
    if len(unique_subsets) == 0:
        logger.warning(f"No subsets found in column '{subset_col}' for dataset '{alias}'.")
        return

    # 5) Determine layout for subplots
    num_subsets = len(unique_subsets)
    if layout == "horizontal":
        num_cols = 2
        num_rows = ceil(num_subsets / num_cols)
        fig, axes = plt.subplots(
            num_rows, num_cols, 
            figsize=(12, 6 * num_rows), 
            sharex=False, 
            sharey=False
        )
    elif layout == "vertical":
        num_cols = 1
        num_rows = num_subsets
        fig, axes = plt.subplots(
            num_rows, num_cols, 
            figsize=(6, 6 * num_subsets), 
            sharex=False, 
            sharey=False
        )
    else:
        logger.error("Invalid layout specified. Use 'horizontal' or 'vertical'.")
        raise ValueError("Invalid layout specified. Use 'horizontal' or 'vertical'.")

    # Flatten axes for easy iteration
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    # Remove any unused subplots if we have fewer subsets than subplot slots
    for i in range(num_subsets, len(axes)):
        fig.delaxes(axes[i])

    # 6) Plot each subset in its own subplot
    for i, subset_val in enumerate(unique_subsets):
        subset_data = data[data[subset_col] == subset_val].copy()

        # Apply order overrides for x_col if provided
        if order_overrides:
            subset_data[x_col] = pd.Categorical(
                subset_data[x_col],
                categories=order_overrides,
                ordered=True
            )
            unique_x = list(subset_data[x_col].cat.categories)
        else:
            unique_x = sorted(subset_data[x_col].unique())

        # Prepare data for boxplot
        boxplot_data = [
            subset_data[subset_data[x_col] == x_val][y_col] 
            for x_val in unique_x
        ]

        ax = axes[i]

        # Basic style options for the boxplot
        boxprops = dict(facecolor="#AED6F1", color="#2C3E50")
        medianprops = dict(color="#2C3E50")
        whiskerprops = dict(color="#2C3E50")
        capprops = dict(color="#2C3E50")
        flierprops = dict(markerfacecolor="#EC7063", markeredgecolor="#2C3E50")

        ax.boxplot(
            boxplot_data,
            patch_artist=True,
            boxprops=boxprops,
            medianprops=medianprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            flierprops=flierprops
        )

        # Set x ticks and labels
        ax.set_xticks(range(1, len(unique_x) + 1))

        # Format numeric x-axis labels if requested
        if format_x_labels:
            if all(isinstance(x_val, (int, float)) for x_val in unique_x):
                formatted_labels = [
                    int(x_val) if x_val == int(x_val) else x_val
                    for x_val in unique_x
                ]
            else:
                formatted_labels = unique_x
        else:
            formatted_labels = unique_x

        ax.set_xticklabels(formatted_labels)

        # Dynamic y-limits
        y_min = subset_data[y_col].min()
        y_max = subset_data[y_col].max()
        ax.set_ylim(y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max))

        # Subplot title shows the subset
        ax.set_title(f"{detailed_name} | {subset_col} = {subset_val}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle='--', alpha=0.5)

    # 7) Layout and optional saving
    fig.suptitle(f"Actual-vs-Fitted Boxplots for {detailed_name}\n(Subsets by '{subset_col}')")
    plt.tight_layout()

    if output_filepath_prefix:
        # Save the entire multi-plot figure as PNG and TikZ
        png_filename = f"{output_filepath_prefix}{alias}_subsets.png"
        tex_filename = f"{output_filepath_prefix}{alias}_subsets.tex"

        fig.savefig(png_filename, dpi=300)
        tikzplotlib.save(tex_filename)
        logger.info(f"Saved multi-subset boxplot to '{png_filename}' and '{tex_filename}'")

    plt.show()

