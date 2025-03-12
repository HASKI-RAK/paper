import matplotlib.pyplot as plt
import pandas as pd
from math import ceil
from loguru import logger
from typing import Optional, Dict, List
import tikzplotlib  

def actual_vs_fitted_boxplots(
    evaluation_data,
    layout: str = "horizontal",
    x_col: str = "human_score",
    y_col: str = "model_score",
    x_label: str = "Human Score",
    y_label: str = "Model Score",
    format_x_labels: bool = True,
    grouping_overrides: Optional[Dict[str, str]] = None,
    order_overrides: Optional[Dict[str, List]] = None,
    include_aliases: Optional[List[str]] = None,
    exclude_aliases: Optional[List[str]] = None,
    filters: Optional[Dict[str, str]] = None,
    output_filepath_prefix: Optional[str] = None
):
    """
    Creates grouped boxplots for datasets in an EvaluationDataClass instance,
    optionally filtering which datasets to include or exclude, and optionally saving
    each subplot as an individual PNG and TEX file.

    If 'output_filepath_prefix' is provided, each boxplot is also saved as both PNG
    and LaTeX (using tikzplotlib). For example, if you pass "./exports/boxplot_", 
    then each dataset alias 'X' will be saved under:
       - "./exports/boxplot_X.png"
       - "./exports/boxplot_X.tex"

    Args:
        evaluation_data (EvaluationDataClass): The EvaluationDataClass object 
                                               containing multiple datasets.
        layout (str): Layout of the plots, either 'horizontal' or 'vertical'.
        x_col (str): Default column name for grouping (X-axis). Defaults to "human_score".
        y_col (str): The column name for the predicted values (Y-axis). Defaults to "model_score".
        x_label (str): Label for the X-axis. Defaults to "Human Score".
        y_label (str): Label for the Y-axis. Defaults to "Model Score".
        format_x_labels (bool): If True, format numeric X-axis labels (integer vs. float). Defaults to True.
        grouping_overrides (dict, optional): If provided, a dict mapping dataset aliases 
                                             to a custom grouping column (instead of x_col).
        order_overrides (dict, optional): If provided, a dict mapping dataset aliases 
                                          to an explicit list defining the category order.
        include_aliases (list, optional): Only plot these dataset aliases. If None, use all unless excluded.
        exclude_aliases (list, optional): Exclude these dataset aliases. If None, exclude none.
        filters (dict, optional): A dictionary mapping dataset aliases to filter conditions. 
                                  Each filter condition should be a string formatted as a 
                                  pandas query expression, applied to the corresponding dataset. 
        output_filepath_prefix (str, optional): If provided, each boxplot is saved 
                                                as a PNG and a TEX (via tikzplotlib). 
                                                Defaults to None.

    Returns:
        None
    """
    # 1) Gather all possible aliases
    all_aliases = list(evaluation_data.datasets_metadata.keys())

    # 2) If include_aliases is specified, limit to those
    if include_aliases is not None:
        aliases_to_plot = [alias for alias in all_aliases if alias in include_aliases]
    else:
        aliases_to_plot = all_aliases

    # 3) If exclude_aliases is specified, remove them from the list
    if exclude_aliases is not None:
        aliases_to_plot = [alias for alias in aliases_to_plot if alias not in exclude_aliases]

    # If there's nothing left to plot, exit early
    num_datasets = len(aliases_to_plot)
    if num_datasets == 0:
        logger.warning("No datasets to plot after applying include/exclude filters.")
        return

    # Determine layout
    if layout == "horizontal":
        num_cols = 2
        num_rows = ceil(num_datasets / num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6 * num_rows), sharex=False, sharey=False)
    elif layout == "vertical":
        num_cols = 1
        num_rows = num_datasets
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(6, 6 * num_rows), sharex=False, sharey=False)
    else:
        logger.error("Invalid layout specified. Use 'horizontal' or 'vertical'.")
        raise ValueError("Invalid layout specified. Use 'horizontal' or 'vertical'.")

    # Flatten axes for easy iteration
    if isinstance(axes, plt.Axes):  
        axes = [axes]
    else:
        axes = axes.flatten()

    # Remove any unused subplots if we have fewer datasets than subplot slots
    for i in range(num_datasets, len(axes)):
        fig.delaxes(axes[i])

    # Create a boxplot for each dataset
    for i, alias in enumerate(aliases_to_plot):
        dataset = evaluation_data.get_dataset(alias)
        if dataset is None:
            logger.warning(f"Dataset '{alias}' not found in EvaluationDataClass.")
            continue

        data = dataset.df.copy()
        metadata = evaluation_data.datasets_metadata[alias]

        # Determine actual grouping column, possibly overridden
        if grouping_overrides and alias in grouping_overrides:
            dataset_x_col = grouping_overrides[alias]
        else:
            dataset_x_col = x_col

        # Check required columns
        if dataset_x_col not in data.columns or y_col not in data.columns:
            logger.error(
                f"Column '{dataset_x_col}' or '{y_col}' not found in dataset '{alias}'. "
                f"Skipping this dataset."
            )
            continue

        # Optionally filter for certain subset
        if filters and alias in filters:
            data = data.query(filters[alias]).copy()

        # Optionally apply an explicit category order
        if order_overrides and alias in order_overrides:
            cat_order = order_overrides[alias]
            data[dataset_x_col] = pd.Categorical(
                data[dataset_x_col],
                categories=cat_order,
                ordered=True
            )
            unique_x = list(data[dataset_x_col].cat.categories)
        else:
            unique_x = sorted(data[dataset_x_col].unique())

        # Organize data for each category
        boxplot_data = [
            data[data[dataset_x_col] == x_val][y_col] 
            for x_val in unique_x
        ]

        # Plot to the correct subplot axis
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

        # Format numeric labels if requested
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
        y_min = data[y_col].min()
        y_max = data[y_col].max()
        ax.set_ylim(y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max))

        detailed_name = metadata.get('detailed_name', alias)
        ax.set_title(f"Actual-vs-Fitted-Boxplots for {detailed_name}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle='--', alpha=0.5)

        # If requested, save each plot individually
        if output_filepath_prefix:
            # Create a separate figure for just this dataset's boxplot
            fig_single, ax_single = plt.subplots(figsize=(6, 6))

            # Repeat the boxplot creation for the single figure
            ax_single.boxplot(
                boxplot_data,
                patch_artist=True,
                boxprops=boxprops,
                medianprops=medianprops,
                whiskerprops=whiskerprops,
                capprops=capprops,
                flierprops=flierprops
            )

            ax_single.set_xticks(range(1, len(unique_x) + 1))
            ax_single.set_xticklabels(formatted_labels)
            ax_single.set_ylim(y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max))
            ax_single.grid(True, linestyle='--', alpha=0.5)
            ax_single.set_title(f"Actual-vs-Fitted-Boxplots for {detailed_name}")
            ax_single.set_xlabel(x_label)
            ax_single.set_ylabel(y_label)

            fig_single.tight_layout()

            # Construct filenames
            filename_png = f"{output_filepath_prefix}{alias}.png"
            filename_tex = f"{output_filepath_prefix}{alias}.tex"

            # Save the PNG
            fig_single.savefig(filename_png, dpi=300)

            # Save the TikZ/LaTeX
            tikzplotlib.save(filename_tex)

            logger.info(f"Saved individual boxplot to '{filename_png}' and '{filename_tex}'")
            plt.close(fig_single)

    # Show the multi-subplot figure in the notebook
    plt.tight_layout()
    plt.show()

