import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from matplotlib.colors import LinearSegmentedColormap
from loguru import logger
from typing import List, Optional
import tikzplotlib

def actual_vs_fitted_heatmaps(
    evaluation_data,
    layout: str = "horizontal",
    y_col: str = "model_score_rounded",
    x_col: str = "human_score",
    vmin=0,
    vmax=20,
    absolute=False,
    diagonal_color="black",
    debug=False,
    include_aliases: Optional[List[str]] = None,
    exclude_aliases: Optional[List[str]] = None,
    output_filepath_prefix: Optional[str] = None
):
    """
    Creates heatmaps for selected datasets managed by EvaluationDataClass to visualize 
    the frequency of each combination of actual (human) and fitted (model) scores,
    either as percentages or absolute values. Optionally highlights the diagonal where 
    human_score == model_score_rounded.

    If 'output_filepath_prefix' is provided, each heatmap is also saved as both PNG and 
    LaTeX (using tikzplotlib). For example, if you pass "./exports/heatmap_", then each 
    dataset alias 'X' will be saved under "./exports/heatmap_X.png" and 
    "./exports/heatmap_X.tex".

    Parameters:
        evaluation_data (EvaluationDataClass): The EvaluationDataClass object containing multiple datasets.
        layout (str): Layout of the plots, either 'horizontal' or 'vertical'.
        y_col (str): The column name for the fitted values. Defaults to "model_score_rounded".
        x_col (str): The column name for the actual values. Defaults to "human_score".
        vmin (float): Minimum value for the color scale. Defaults to 0.
        vmax (float): Maximum value for the color scale. Defaults to 20 for percentages. 
                      If `absolute` is True, `vmax` should be adjusted accordingly.
        absolute (bool): If True, displays absolute values instead of percentages. Defaults to False.
        diagonal_color (str): Color to highlight the diagonal. Defaults to "black".
        debug (bool): If True, prints intermediate DataFrame results for debugging. Defaults to False.
        include_aliases (list, optional): List of dataset aliases to explicitly include. 
                                          If None, all datasets will be considered unless excluded.
        exclude_aliases (list, optional): List of dataset aliases to explicitly exclude. 
                                          If None, no datasets will be excluded (unless `include_aliases` is given).
        output_filepath_prefix (str, optional): If provided, each heatmap is also saved 
                                                as a PNG and a TEX (via tikzplotlib). 
                                                Defaults to None.

    Returns:
        None
    """
    # 1) Collect all dataset aliases from evaluation_data
    all_aliases = list(evaluation_data.datasets_metadata.keys())

    # 2) If include_aliases is provided, reduce the list to those
    if include_aliases is not None:
        aliases_to_plot = [alias for alias in all_aliases if alias in include_aliases]
    else:
        aliases_to_plot = all_aliases

    # 3) If exclude_aliases is provided, remove those from the final list
    if exclude_aliases is not None:
        aliases_to_plot = [alias for alias in aliases_to_plot if alias not in exclude_aliases]

    # At this point, aliases_to_plot contains only the datasets we want
    num_datasets = len(aliases_to_plot)
    if num_datasets == 0:
        logger.warning("No datasets to plot after applying include/exclude filters.")
        return
    
    # Configure subplot layout
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

    # Flatten the axes array for easy iteration
    if isinstance(axes, plt.Axes):  # Single subplot case
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Remove unused subplots
    for i in range(num_datasets, len(axes)):
        fig.delaxes(axes[i])
        
    # Define a custom colormap with a focus on lower and higher values
    colors = [
        (0.0, "#FFFFFF"),  # 0% -> white
        (0.05, "#2E86C1"), # 1–5% -> blue
        (1.0, "#C0392B")   # up to 100% -> red
    ]
    cmap = LinearSegmentedColormap.from_list("Custom", colors, N=256)

    # Create heatmaps for each dataset alias
    for i, alias in enumerate(aliases_to_plot):
        dataset = evaluation_data.get_dataset(alias)
        if dataset is None:
            logger.warning(f"Dataset '{alias}' not found in EvaluationDataClass.")
            continue
        data = dataset.df

        if x_col not in data.columns or y_col not in data.columns:
            logger.error(f"Column '{x_col}' or '{y_col}' not found in dataset '{alias}'. Skipping this dataset.")
            continue

        ax = axes[i]

        # Create a pivot table for the heatmap
        heatmap_data = data.pivot_table(
            index=y_col, 
            columns=x_col, 
            aggfunc='size', 
            fill_value=0
        ).sort_index(ascending=False)

        if debug:
            print(f"Debug pivot table for dataset '{alias}':")
            print(heatmap_data)

        if absolute:
            heatmap_data_to_plot = heatmap_data
            colorbar_label = "Absolute Count"
        else:
            # Convert counts to percentages
            heatmap_data_to_plot = heatmap_data / heatmap_data.values.sum() * 100
            colorbar_label = "Percentage (%)"

        # Reindex to ensure 0-based ordering on both axes
        heatmap_data_to_plot = heatmap_data_to_plot.reindex(
            index=sorted(heatmap_data_to_plot.index),
            columns=sorted(heatmap_data_to_plot.columns)
        )

        sns.heatmap(
            heatmap_data_to_plot,
            annot=True,
            fmt=".1f" if not absolute else "d",  # show one decimal place for percentages
            cmap=cmap,
            cbar=True,
            cbar_kws={'label': colorbar_label},
            vmin=vmin,
            vmax=vmax,
            ax=ax
        )

        # Highlight diagonal cells
        for j in range(len(heatmap_data_to_plot.index)):
            if j < len(heatmap_data_to_plot.columns):
                ax.add_patch(plt.Rectangle((j, j), 1, 1, fill=False, 
                                           edgecolor=diagonal_color, lw=2))

        detailed_name = evaluation_data.datasets_metadata[alias].get('detailed_name', alias)
        ax.set_title(f"Actual-vs-Fitted Heatmap for {detailed_name}")
        ax.set_xlabel('Human Score')
        ax.set_ylabel('Model Score (Rounded)')
        ax.invert_yaxis()  
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

        # Save individual plots (PNG + TEX) if a prefix was provided
        if output_filepath_prefix:
            # Create a separate figure for just this heatmap
            fig_single, ax_single = plt.subplots(figsize=(6, 6))
            
            sns.heatmap(
                heatmap_data_to_plot,
                annot=True,
                fmt=".1f" if not absolute else "d",
                cmap=cmap,
                cbar=True,
                cbar_kws={'label': colorbar_label},
                vmin=vmin,
                vmax=vmax,
                ax=ax_single
            )
            
            # Highlight diagonal cells in the single figure
            for j in range(len(heatmap_data_to_plot.index)):
                if j < len(heatmap_data_to_plot.columns):
                    ax_single.add_patch(plt.Rectangle((j, j), 1, 1, fill=False, 
                                                      edgecolor=diagonal_color, lw=2))
            
            ax_single.set_title(f"Actual-vs-Fitted Heatmap for {detailed_name}")
            ax_single.set_xlabel('Human Score')
            ax_single.set_ylabel('Model Score (Rounded)')
            ax_single.invert_yaxis()  
            ax_single.tick_params(axis='x', rotation=45)
            ax_single.tick_params(axis='y', rotation=0)
            
            fig_single.tight_layout()

            # Construct filenames
            filename_png = f"{output_filepath_prefix}{alias}.png"
            filename_tex = f"{output_filepath_prefix}{alias}.tex"

            # Save the PNG
            fig_single.savefig(filename_png, dpi=300)
            
            # Save the TikZ/LaTeX
            tikzplotlib.save(filename_tex)
            
            logger.info(f"Saved individual heatmap to '{filename_png}' and '{filename_tex}'")
            plt.close(fig_single)

    # Show the multi-subplot figure in the notebook
    plt.tight_layout()
    plt.show()
