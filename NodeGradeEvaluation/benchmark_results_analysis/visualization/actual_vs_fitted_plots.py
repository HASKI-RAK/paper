import matplotlib.pyplot as plt
import tikzplotlib
from math import ceil
from loguru import logger
from typing import List, Optional
from src import EvaluationDataClass

def actual_vs_fitted_plots(
    evaluation_data: EvaluationDataClass, 
    x_col: str = 'human_score',
    y_col: str = 'model_score',
    x_label: str = 'Human Score',
    y_label: str = 'Model Score',
    layout: str = "horizontal", 
    dot_color: str = "#2C3E50",
    cmap_str: str = 'cividis',
    group_by: Optional[str] = None,
    legend_title: Optional[str] = None,
    include_aliases: Optional[List[str]] = None,
    exclude_aliases: Optional[List[str]] = None,
    output_filepath_prefix: Optional[str] = None
):
    """
    Creates Actual-vs-Fitted-Plots for datasets managed by an EvaluationDataClass instance.
    Optionally filters which datasets to include or exclude, and optionally saves each subplot 
    as an individual PNG and TEX file if 'output_filepath_prefix' is provided.

    If 'output_filepath_prefix' is provided, each scatter plot is also saved as both PNG
    and LaTeX (using tikzplotlib). For example, if you pass "./exports/scatter_", 
    then each dataset alias 'X' will be saved under:
       - "./exports/scatter_X.png"
       - "./exports/scatter_X.tex"

    Args:
        evaluation_data (EvaluationDataClass): The EvaluationDataClass object containing multiple datasets.
        x_col (str): Column name for the true score (X-axis). Defaults to 'human_score'.
        y_col (str): Column name for the model-predicted score (Y-axis). Defaults to 'model_score'.
        x_label (str): Label for the X-axis. Defaults to 'Human Score'.
        y_label (str): Label for the Y-axis. Defaults to 'Model Score'.
        layout (str): Layout of the plots, either 'horizontal' or 'vertical'. Defaults to 'horizontal'.
        dot_color (str): Color of the dots in the scatter plots (ignored if 'group_by' is used). Defaults to '#2C3E50'.
        group_by (Optional[str]): Column name for color-grouping the points. Defaults to None.
        legend_title (Optional[str]): Custom title for the legend. If None, uses the group_by column name.
        include_aliases (Optional[List[str]]): Only plot these dataset aliases. If None, use all unless excluded.
        exclude_aliases (Optional[List[str]]): Exclude these dataset aliases. If None, exclude none.
        output_filepath_prefix (Optional[str]): If provided, saves each scatter plot as PNG and TEX. Defaults to None.

    Returns:
        None
    """

    # 1) Gather all possible aliases
    all_aliases = list(evaluation_data.datasets_metadata.keys())

    # 2) If include_aliases is specified, limit to those
    if include_aliases is not None:
        aliases_to_use = [alias for alias in all_aliases if alias in include_aliases]
    else:
        aliases_to_use = all_aliases

    # 3) If exclude_aliases is specified, remove them from the list
    if exclude_aliases is not None:
        aliases_to_use = [alias for alias in aliases_to_use if alias not in exclude_aliases]

    # If there's nothing left to plot, exit early
    num_datasets = len(aliases_to_use)
    if num_datasets == 0:
        logger.warning("No datasets to plot after applying include/exclude filters.")
        return

    # 4) Determine layout for the multi-subplot figure
    if layout == "horizontal":
        num_cols = 2
        num_rows = ceil(num_datasets / num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6 * num_rows), sharex=True, sharey=True)
    elif layout == "vertical":
        num_cols = 1
        num_rows = num_datasets
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(6, 6 * num_rows), sharex=True, sharey=True)
    else:
        raise ValueError("Invalid layout specified. Use 'horizontal' or 'vertical'.")

    # 5) Flatten the axes array for easy iteration
    if layout == "horizontal":
        axes = axes.flatten()
    else:
        axes = [axes] if num_datasets == 1 else axes.flatten()

    # Ensure axes is a list (it might be a single Axes if only one plot)
    if num_datasets == 1:
        axes = [axes]

    # 6) Create a scatter plot for each dataset
    for i, alias in enumerate(aliases_to_use):
        dataset = evaluation_data.get_dataset(alias)
        if dataset is None:
            logger.warning(f"Dataset '{alias}' not found in EvaluationDataClass.")
            continue

        data = dataset.df
        metadata = evaluation_data.datasets_metadata[alias]
        ax = axes[i]

        # If group_by is provided, we create a color-coded scatter plot
        if group_by and group_by in data.columns:
            unique_groups = data[group_by].unique()
            cmap = plt.colormaps.get_cmap(cmap_str)
            color_map = {group: cmap(j) for j, group in enumerate(unique_groups)}

            for group in unique_groups:
                subset = data[data[group_by] == group]
                ax.scatter(subset[x_col], subset[y_col], alpha=0.6, label=str(group), color=color_map[group])
            ax.legend(title=legend_title if legend_title else group_by)
        else:
            # Simple scatter without color grouping
            ax.scatter(data[x_col], data[y_col], alpha=0.6, color=dot_color)
            # If no group_by, you might not want a legend at all. 
            # But to mimic the existing code, we keep the same logic:
            ax.legend(title=legend_title if legend_title else group_by)

        # Set labels and title
        ax.set_title(f"Actual-vs-Fitted-Plot {metadata['detailed_name']}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # 7) If requested, save each plot individually
        if output_filepath_prefix:
            fig_single, ax_single = plt.subplots(figsize=(6, 6))

            if group_by and group_by in data.columns:
                for group in unique_groups:
                    subset = data[data[group_by] == group]
                    ax_single.scatter(subset[x_col], subset[y_col], alpha=0.6, 
                                      label=str(group), color=color_map[group])
                ax_single.legend(title=legend_title if legend_title else group_by)
            else:
                ax_single.scatter(data[x_col], data[y_col], alpha=0.6, color=dot_color)
                ax_single.legend(title=legend_title if legend_title else group_by)

            ax_single.set_title(f"Actual-vs-Fitted-Plot {metadata['detailed_name']}")
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

            logger.info(f"Saved individual scatter plot to '{filename_png}' and '{filename_tex}'")
            plt.close(fig_single)

    # 8) Remove any unused subplots if fewer datasets than subplot slots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # 9) Show the multi-subplot figure in the notebook
    plt.tight_layout()
    plt.show()
