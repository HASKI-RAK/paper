# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: ASAG Evaluation venv
#     language: python
#     name: asag_venv
# ---

# <h1 style="
#     font-family: 'Times New Roman', Times, serif;
#     font-size: 36px;
#     color: #2C3E50;
#     text-align: center;
#     border-bottom: 2px solid #BDC3C7;
#     padding-bottom: 10px;
#     margin-bottom: 20px;
# ">
#     Evaluation of NodeGrade on Public Datasets
# </h1>
#

# +
# libraries
import os
import re
from loguru import logger
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.legend import Legend

import seaborn as sns

# add missing attributes (monkey patch for compatibility with tikzplotlib)
Line2D._us_dashSeq = property(lambda self: self._dash_pattern[1])
Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])
Legend._ncol = property(lambda self: self._ncols)
import tikzplotlib
import numpy as np
import json
import sys

# custom imports
from src import EvaluationDataClass

from helpers import (
    scale_column,
    check_ranges,
    performance_metrics,
    calculate_kappa_scores,
    evaluate_classification,
    optimize_threshold,
    plot_metric_vs_threshold,
    customized_html_export
)

from visualization import (
    actual_vs_fitted_plots,
    actual_vs_fitted_boxplots,
    actual_vs_fitted_heatmaps
)


flags = {}
os.makedirs("./exports", exist_ok=True)

# -

# <h2 style="
#     font-family: 'Times New Roman', Times, serif;
#     font-size: 28px;
#     color: #2C3E50;
#     text-align: left;
#     border-bottom: 1px solid #BDC3C7;
#     padding-bottom: 5px;
#     margin-top: 30px;
#     margin-bottom: 10px;
# ">
#     Configurations
# </h2>
#

# Configure logger
logger.remove()
_=logger.add(sys.stderr, level="INFO")

# +
# Configure alias, detailed name, filepaths and configurations regarding 
# data preparation for each dataset 

datasets_metadata = {
    'mohler_en': {
        'filepath': './data/ng_mohler-20250123192635_en.json',
        'detailed_name': 'Mohler (english prompts)',
        'scaling_model_score': {
            'perform_scaling': True,
            'new_min':0.0,
            'new_max':5.0,
            'old_min':0.0,
            'old_max':100.0
        },
        'rounding_step': 0.5
    },

    'mohler_de': {
        'filepath': './data/ng_mohler-20250123213529_de.json',
        'detailed_name': 'Mohler (german prompts)',
        'scaling_model_score': {
            'perform_scaling': True,
            'new_min':0.0,
            'new_max':5.0,
            'old_min':0.0,
            'old_max':100.0
        },
        'rounding_step': 0.5
    },
    
    'engsaf': {
        'filepath': './data/ng_engsaf-20250123041225.json',
        'detailed_name':  'Engineering Short Answer Feedback',
        'scaling_model_score': {
            'perform_scaling': True,
            'new_min':0.0,
            'new_max':2.0,
            'old_min':0.0,
            'old_max':100.0
        },
        'rounding_step': 1.0        
    },

    'semeval': {
        'filepath': './data/ng_semeval-20250129_en.json',
        'detailed_name':  'SemEval 2013',
        'scaling_model_score': {
            'perform_scaling': True,
            'new_min':0.0,
            'new_max':1.0,
            'old_min':0.0,
            'old_max':100.0
        },
        'rounding_step': 1.0,
        'flatten_json_in_results': ['metadata']
    },

    'os_en': {
        'filepath': './data/ng_os-20250129210935.json',
        'detailed_name': 'OS Dataset (english prompts)',
        'scaling_model_score': {'perform_scaling': False},
        'flatten_json_in_results': ['metadata']
    },

    'os_de': {
        'filepath': './data/ng_os-20250130000947.json',
        'detailed_name':  'OS Dataset (german prompts)',
        'scaling_model_score': {'perform_scaling': False},
        'flatten_json_in_results': ['metadata']
    }
}
# -

# <h2 style="
#     font-family: 'Times New Roman', Times, serif;
#     font-size: 28px;
#     color: #2C3E50;
#     text-align: left;
#     border-bottom: 1px solid #BDC3C7;
#     padding-bottom: 5px;
#     margin-top: 30px;
#     margin-bottom: 10px;
# ">
#     Load Data
# </h2>
#

# +
# Load data
evaluation_data = EvaluationDataClass(
    datasets_metadata=datasets_metadata
)

# Check ranges regarding human_score and model_score
check_ranges(
    evaluation_data,
    ['human_score', 'model_score'],
    output_mode = 'return_df'
)
# -

# <h2 style="
#     font-family: 'Times New Roman', Times, serif;
#     font-size: 28px;
#     color: #2C3E50;
#     text-align: left;
#     border-bottom: 1px solid #BDC3C7;
#     padding-bottom: 5px;
#     margin-top: 30px;
#     margin-bottom: 10px;
# ">
#     Data Preparation
# </h2>
#

# +
dimtables = {}

# Data-Prepration
for alias in datasets_metadata.keys():
    # get reference of dataset
    dataset = evaluation_data.get_dataset(alias)

    # set negative values in model_score to zero
    dataset.df.loc[dataset.df['model_score'] < 0, 'model_score'] = 0

    # Rescale model_score, if specified
    perform_scaling = datasets_metadata.get(alias, {}).get('scaling_model_score', {}).get('perform_scaling', False)
    
    if perform_scaling:
        # save original model_score without negative values, but before scaling
        dataset.df['model_score_original'] = dataset.df['model_score']
        dataset.df = scale_column(
            df=dataset.df,
            column_name='model_score',
            new_min=datasets_metadata[alias]['scaling_model_score']['new_min'],
            new_max=datasets_metadata[alias]['scaling_model_score']['new_max'],
            old_min=datasets_metadata[alias]['scaling_model_score']['old_min'],
            old_max=datasets_metadata[alias]['scaling_model_score']['old_max']
        )   
        logger.info(f"\nScaled variables 'model_score' for dataset '{alias}'")
    else:
        logger.info(f"\nSkipping scaling for dataset {alias}.")

    # Special data preparation for OS Datasets
    if alias in ['os_en', 'os_de']:
        # drop duplicates and log
        n_rows = dataset.df.shape[0]
        print(n_rows)
        dataset.df = dataset.df.drop_duplicates(subset=[
           'model_score', 
            'human_score', 
            'student_answer', 
            'instructor_answer',
            'question'
        ], keep = 'first')
        n_dropped_duplicates = n_rows - dataset.df.shape[0]
        if n_dropped_duplicates > 0:
            logger.info(f"Dropped {n_dropped_duplicates} duplicates (Alias: {alias})")
        else:
            logger.info(f"Dropped no duplicates (Alias: {alias})")
        
        # standardize human_score (each question differs in its possible maximum score)
        dataset.df['human_score_zero_hundred_scaled'] = (
            dataset.df['human_score'] / dataset.df['metadata_full_points']
        ) * 100        

        # retain original model_score
        dataset.df['model_score_original'] = dataset.df['model_score'] 

        # Rescale model_score to human_score scale
        dataset.df['model_score'] = (
            dataset.df['model_score'] / 100
        ) * dataset.df['metadata_full_points']

        # round model score to nearest absolute number
        dataset.df['model_score_rounded'] = round(dataset.df['model_score'])

        # build question id based on github folder names
        dataset.df['question_id_2'] = dataset.df['question_id'].str[0].astype(int)

        # get unique questions and build relation table
        question_id_dimension_table = evaluation_data.datasets[alias].df.filter([
            'instructor_answer',
            'metadata_full_points',
            'question_id_2'
        ]).drop_duplicates().reset_index(drop=True).assign(custom_question_id=lambda df: range(1, len(df) + 1))
    
        # build and save relation table
        dimtables[alias] = question_id_dimension_table
        question_id_dimension_table.to_csv(f'./exports/dim_table_custom_question_id_{alias}.csv')
        
        # update dataset
        evaluation_data.datasets[alias].df = evaluation_data.datasets[alias].df.merge(
            question_id_dimension_table.drop(columns=['question_id_2']),
            on=['instructor_answer', 'metadata_full_points'],
            how='left'
        )
        
        # log action
        msg = f"Performed special data preparation for dataset {alias}"
        logger.info(msg)
        # set flag that os dataset was prepared
        flags['prepared_os_data'] = True

 
    # Apply rounding based on dataset-specific configurations to achieve a
    # model_score that aligns with the discrete steps of the corresponding
    # human_score scale, ensuring consistency between predicted and actual scoring systems.
    rounding_step = datasets_metadata.get(alias, {}).get('rounding_step', None)
    if rounding_step is not None:
        dataset.df['model_score_rounded'] = dataset.df['model_score'].apply(
            lambda x: round(x / rounding_step) * rounding_step
        )
        logger.info(f"\nRounded 'model_score' to nearest rounding step ({rounding_step}) for dataset '{alias}' and stored the result in new column named 'model_score_rounded'")
    else:
        logger.info(f"\nNo rounding_step defined for dataset '{alias}'. Skipping rounding.")

if not flags['prepared_os_data']:
    logger.warn("OS-Dataset-Handling Flag not set. Changed alias? -> Problem! Removed dataset? -> Calm down!")

# QS: Check ranges
pd.concat([
    check_ranges(
        evaluation_data,
        ['human_score', 'model_score', 'model_score_rounded', 'model_score_original'],
        output_mode = 'return_df',
        exclude_aliases = ['os_en', 'os_de']
    ),
    
    check_ranges(
        evaluation_data,
        ['human_score', 'model_score', 'model_score_rounded', 'human_score_zero_hundred_scaled'],
        output_mode = 'return_df',
        include_aliases = ['os_en', 'os_de']
    )
])


# -

# <h2 style="
#     font-family: 'Times New Roman', Times, serif;
#     font-size: 28px;
#     color: #2C3E50;
#     text-align: left;
#     border-bottom: 1px solid #BDC3C7;
#     padding-bottom: 5px;
#     margin-top: 30px;
#     margin-bottom: 10px;
# ">
#     Visual Analysis
# </h2>
#

actual_vs_fitted_boxplots(
    evaluation_data=evaluation_data,
    format_x_labels=False,
    layout="horizontal",
    y_col="model_score",
    x_col="human_score",
    grouping_overrides={"semeval": "metadata_original_label"},
    order_overrides={"semeval": ["incorrect", "contradictory", "correct"]},
    output_filepath_prefix = './exports/fitvsact_boxplot_',
    exclude_aliases=["os_en", "os_de"]
)


# Für Semeval steht neben dem Merkmal metadata_original_label auch ein binäres Grading zur Verfügung:
actual_vs_fitted_boxplots(
    evaluation_data=evaluation_data,
    format_x_labels=False,
    layout="horizontal",
    y_col="model_score",
    x_col="human_score",
    include_aliases = ['semeval'],
    output_filepath_prefix = './exports/fitvsact_boxplot_binary_',
)

actual_vs_fitted_heatmaps(
    evaluation_data,
    y_col="model_score_rounded",
    x_col="human_score",
    absolute = False,
    no_title=True,
    vmin=0,
    vmax=20,
    diagonal_color='green',
    output_filepath_prefix="./exports/heatmap_",
    exclude_aliases=["os_en", "os_de"]
)

# +
# ensure corrrect sorting
for al in ['os_en', 'os_de']:
    evaluation_data.datasets[al].df = evaluation_data.datasets[al].df.sort_values(by='question_id_2')

actual_vs_fitted_plots(
    evaluation_data = evaluation_data, 
    x_col = 'human_score',
    y_col = 'model_score',
    x_label = 'Human Score',
    y_label = 'Model Score',
    layout = "horizontal", 
    group_by='question_id_2',
    legend_title=" ",#"Question ID",
    no_title=True, # False
    include_aliases = ['os_en', 'os_de'],
    output_filepath_prefix = './exports/scatter_contin_',
    cmap_str = "Dark2" 
)
# -

# __Whats's up with question 6?__

for i in range(5):
    example_row = evaluation_data.datasets['os_en'].df.query(
        'question_id_2 == 6'
    ).filter(['question', 'instructor_answer']).drop_duplicates().values[i]

    print(f"------------\nExample  {i}\n-----------")
    print("Question")
    print(f"\n{example_row[0]}\n")
    print("Instructor Answer")
    print(f"\n{example_row[1]}\n")


actual_vs_fitted_plots(
    evaluation_data = evaluation_data, 
    x_col = 'human_score',
    y_col = 'model_score_rounded',
    x_label = 'Human Score',# (original)',
    y_label = 'Model Score', #(scaled to the range of the human score and discretized)',
    layout = "horizontal", 
    group_by='question_id_2',
    legend_title="Question ID",
    no_title=True,
    include_aliases = ['os_en', 'os_de'],
    output_filepath_prefix = './exports/scatter_discr_',
    cmap_str = "Dark2" 
)

actual_vs_fitted_plots(
    evaluation_data = evaluation_data, 
    x_col = 'human_score',
    y_col = 'model_score_original',
    x_label = 'Human Score (original)',
    y_label = 'Model Score (original)',
    layout = "horizontal", 
    group_by='question_id_2',
    legend_title="Question ID",
    include_aliases = ['os_en', 'os_de'],
    cmap_str = "Dark2"
)

actual_vs_fitted_plots(
    evaluation_data = evaluation_data, 
    x_col = 'human_score_zero_hundred_scaled',
    y_col = 'model_score_original',
    x_label = 'Human Score (scaled to model_score range)',
    y_label = 'Model Score (Original)',
    layout = "horizontal", 
    group_by='question_id_2',
    legend_title="Question ID",
    include_aliases = ['os_en', 'os_de'],
    cmap_str = "Dark2"
)

# <h2 style="
#     font-family: 'Times New Roman', Times, serif;
#     font-size: 28px;
#     color: #2C3E50;
#     text-align: left;
#     border-bottom: 1px solid #BDC3C7;
#     padding-bottom: 5px;
#     margin-top: 30px;
#     margin-bottom: 10px;
# ">
#     Formal Analysis
# </h2>
#
#
# * $\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
#
# * $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$
#
#
# * __$R^2$__
# > Total Sum of Squares
# > 
# > $ \text{SST} = \sum_{i=1}^{n} (Y_i - \overline{Y})^2$
# > 
# > Residual Sum of Squares
# > 
# > $ \text{SSR} = \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2 $
# > 
# > Anteil erklärter Varianz:
# > 
# > $ R^2 = 1 - \frac{\text{SSR}}{\text{SST}} $
#
# * $\text{Precision} = \frac{TP}{TP + FP}$
#
# * $\text{Recall} = \frac{TP}{TP + FN}$
#
# * $\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
#
# * $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$
#
# * $\text{Specificity} = \frac{TN}{TN + FP}$
#
# * $\text{FPR} = \frac{FP}{FP + TN}$
#
# * $\text{FNR} = \frac{FN}{FN + TP}$
#
# * $\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$
#

# <h3 style="
#     font-family: 'Times New Roman', Times, serif;
#     font-size: 22px;
#     color: #2C3E50;
#     text-align: left;
#     border-bottom: 1px dotted #BDC3C7;
#     padding-bottom: 3px;
#     margin-top: 20px;
#     margin-bottom: 10px;
# ">
#     Regresion metrics
# </h3>
#

# +
y_hat = 'model_score'
y = 'human_score'
get_this_metrics = ["RMSE", "MAE", "Pearson"]

# rowbind results
metrics = pd.concat([
    # calculate metrics for all datasets
    performance_metrics(
        evaluation_data,
        y_col = y,
        y_hat_col = y_hat,
        metrics = get_this_metrics
    ),
    # special call regarding Dataset SemEval I (Filter for corpus 'sciEntsBank')
    performance_metrics(
        evaluation_data,
        y_col = y,
        y_hat_col = y_hat,
        include_aliases = 'semeval',
        metrics = get_this_metrics,
        filters = {"semeval": "metadata_corpus == 'sciEntsBank'"}
    ),
    # special call regarding Dataset SemEval II (Filter for corpus 'beetle')
     performance_metrics(
        evaluation_data,
        y_col = y,
        y_hat_col = y_hat,
        include_aliases = 'semeval',
        metrics = get_this_metrics,
        filters = {"semeval": "metadata_corpus == 'beetle'"}
    )
])

del y, y_hat, get_this_metrics
metrics
# -


performance_metrics(
    evaluation_data,
    y_col = "human_score_zero_hundred_scaled",
    y_hat_col = "model_score_original",
    include_aliases = ['os_en', 'os_de'],
    metrics = ["RMSE", "MAE", "R2", "Pearson"],
)

# +
special_results_os_dataset = {}

for alias in ['os_en', 'os_de']:

    # create an empty dataframe
    results = metrics.query("Pearson == 'i need an empty dataframe'")
    
    question_ids = np.sort(evaluation_data.datasets[alias].df['question_id_2'].unique())
    
    
    for cur_id in question_ids:    
        new_entry = performance_metrics(
            evaluation_data,
            y_col = 'human_score',
            y_hat_col = 'model_score',
            metrics =  ["RMSE", "MAE", "Pearson"],
            include_aliases = alias,
            filters = {alias: f"question_id_2 == {cur_id}"}
        )
        
        results = pd.concat([results, new_entry])

        special_results_os_dataset[alias] = results
# -

special_results_os_dataset['os_en'].sort_values(by="Pearson", ascending = False)

special_results_os_dataset['os_de'].sort_values(by="Pearson", ascending = False)

# <h3 style="
#     font-family: 'Times New Roman', Times, serif;
#     font-size: 22px;
#     color: #2C3E50;
#     text-align: left;
#     border-bottom: 1px dotted #BDC3C7;
#     padding-bottom: 3px;
#     margin-top: 20px;
#     margin-bottom: 10px;
# ">
#     Further metrics (dataset specific calculations)
# </h3>

formal_results = {}

# <h4 style="
#     font-family: 'Times New Roman', Times, serif;
#     font-size: 18px;
#     color: #2C3E50;
#     text-align: left;
#     border-bottom: 1px dotted #BDC3C7;
#     padding-bottom: 3px;
#     margin-top: 18px;
#     margin-bottom: 8px;
# ">
#     EngSAF
# </h4>
#
#
# EngSAF verfügt über ein Label mit drei Kategorien: `correct`, `contradicting` `incorrect`
#
# Das bedeutet:
# * Regressionmaße a la $R^2$, RMSE, MAE oder Pearsons $r$ in meinen Augen Unsinn.
# * Weighted Cohens Kappa zwischen ordinal transformierter Vorhersage und True-Label sinnig.
# * ROC-AUC-Werte können mit etwas Kniff für jede Klasse separat berechnet werden.

# +
# specify alias
alias = 'engsaf'

# initialize results dictionairy
formal_results[alias] = {}

# # copy dataset
df = evaluation_data.datasets[alias].df.copy()

# specify and prepare model and human_scores
model_scores = df['model_score_rounded'].astype(int) # classes: 0, 1, 2 ()
human_scores = df['human_score'].astype(int) # classes: 0, 1, 2
# -

# __Cohens Kappa__

# calculate Kappa-Scores
formal_results[alias]['kappa'] = calculate_kappa_scores(
    human_scores,
    model_scores
)

# __ROC-Curve for each class__

# +
distinct_human_scores = df['human_score'].unique()      
formal_results[alias]['MultiClass_ROC_AUC'] = {}
formal_results[alias]['MultiClass_ROC_AUC']['values'] = {}

for value in distinct_human_scores:
    # create dummies and labels for plots
    value_label = None
    if int(value) == 0:
        dummy = (df['human_score'] != value).astype(int)
        value_label = 'Incorrect'
    if int(value) == 1:
        dummy = (df['human_score'] != value).astype(int)
        value_label = 'Partially Correct'
    elif int(value) == 2:
        dummy = (df['human_score'] == value).astype(int)
        value_label = 'Correct'

    # calculate roc auc
    if int(value) in [0, 2]: 
        model_score_original = df['model_score_original']            
        roc_auc = roc_auc_score(dummy, model_score_original)
    elif int(value) == 1: 
        model_score_original = abs(df['model_score_original'] - 50)
        roc_auc = roc_auc_score(dummy, model_score_original)
        
    formal_results[alias]['MultiClass_ROC_AUC']['values'][f'class_{int(value)}'] = float(roc_auc)

    # Calculate the ROC-Curve
    fpr, tpr, thresholds = roc_curve(dummy, model_score_original)

    # plot der ROC-Kurve
    plt.plot(fpr, tpr, label=f'{value_label} (AUC = {roc_auc:.2f})')

# Plot
filename = 'engsaf_roc_curves'
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing') 
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('')
plt.legend(loc='lower right')
plt.grid()
plt.savefig(f"./exports/{filename}.png")
tikzplotlib.save(f"./exports/{filename}.tex")
plt.show()

del roc_auc
# -

# __Calculation of weighted ROC-AUC over all clases__
#
# The weighted ROC-AUC is computed as a weighted sum of the per-class ROC-AUC values, where the weights correspond to the relative frequencies of each class in the dataset.
#
# $\text{Weighted ROC-AUC} = \sum_{i=0}^{N} p_i \cdot \text{ROC-AUC}_i$
#
# where:
# - $N$ is the number of classes,
# - $p_i$ is the proportion of samples in class \( i \),
# - $\text{ROC-AUC}_i$ is the ROC-AUC value for class \( i \).
#

freqs = df['human_score'].value_counts(normalize=True)
freqs

formal_results[alias]['MultiClass_ROC_AUC']['values']

# +
formal_results[alias]['MultiClass_ROC_AUC']['weighted_roc_auc'] = (
    freqs[0.0] * formal_results[alias]['MultiClass_ROC_AUC']['values']['class_0'] +
    freqs[1.0] * formal_results[alias]['MultiClass_ROC_AUC']['values']['class_1'] +
    freqs[2.0] * formal_results[alias]['MultiClass_ROC_AUC']['values']['class_2']
)

print(f"Weighted ROC-AUC: {formal_results[alias]['MultiClass_ROC_AUC']['weighted_roc_auc']:.4f}")
# -

# <h4 style="
#     font-family: 'Times New Roman', Times, serif;
#     font-size: 18px;
#     color: #2C3E50;
#     text-align: left;
#     border-bottom: 1px dotted #BDC3C7;
#     padding-bottom: 3px;
#     margin-top: 18px;
#     margin-bottom: 8px;
# ">
#     Mohler
# </h4>
#
# Mohler verfügt über ein quasiintervallskaliertes True-Label. Das bedeutet:
#
# * Regressionsmaße a la $R^2$, RMSE, MAE oder Pearsons $r$ sinnvoll mit gewissen Einschränkungen, da nur quasi-intervallskalierte True-Labels.
# * Weighted Cohens Kappa zwischen ordinal transformierter Vorhersage und True-Label sinnig.
# * ROC-AUC-Werte in meinen Augen nicht sinnvoll berechenbar.

# __Regressionsmaße__

# calculations from chapter above
metrics.iloc[:2:]

# __Cohens Weighted Kappa__

# +
qs = []

# specify alias
for alias in ['mohler_en', 'mohler_de']:
    print(f"\nAlias: {alias}")
    # initialize results dictionairy
    formal_results[alias] = {}
    
    # copy dataset
    df = evaluation_data.datasets[alias].df.copy()
    
    # specify and prepare model and human_scores     
    # multiplication with 2 because Cohen Kappa needs integer values
    # and is not suitable with decimals
    model_scores = (df['model_score_rounded'] * 2).astype(int)
    human_scores = (df['human_score'] * 2).astype(int)
    
    # qs to check that bevor using int() model_score consists out of natural numbers
    qs.append({
        'model_score_rounded': np.sort((df['model_score_rounded'] * 2).unique()),
        'human_score': np.sort((df['human_score'] * 2).unique())
    })

    # Calculate Kappa Scores
    formal_results[alias]['kappa'] = calculate_kappa_scores(
        human_scores, model_scores
    )

print('\n')
for el in qs: print(el)

# -

# __Checking high discrepance cases__

# +
alias = 'mohler_en'

# select relevant columns
df = evaluation_data.datasets[alias].df.filter([
    'model_score_original',
    'model_score_rounded',
    'model_score',
    'human_score',
    'question',
    'instructor_answer',
    'student_answer'
])

# calculate prediction error (absolute and original)
df['prediction_error'] = (df['model_score'] - df['human_score'])
df['absolute_prediction_error'] = abs(df['prediction_error'])

# mean prediction error per question
df_error_per_ques = df.groupby('question')['prediction_error'].mean().reset_index().sort_values(
    by = 'prediction_error',
    ascending=True)

# Liste der gewünschten Perzentile
percentiles = [75, 80, 85, 90]

# Für jedes Perzentil eine neue Spalte hinzufügen
for p in percentiles:
    col_name = f'percentile_{p}'
    df[col_name] = df.groupby('human_score')['absolute_prediction_error'].transform(
        lambda x: x.quantile(p / 100)
    )


# calculate percentile for each group of 'human_score' and save it as a new column
df_discrepancy = df

df_discrepancy['percentile_rank_grouped'] = df_discrepancy.groupby('human_score')['absolute_prediction_error'].transform(
    lambda x: x.rank(pct=True)
)

df_discrepancy.query("percentile_rank_grouped > 0.75")
# alternatively:
# df_discrepancy.query("absolute_prediction_error > percentile_75") 

# -


for var in ['prediction_error', 'absolute_prediction_error']:
    mean_error = df_discrepancy[var].mean()
    plt.figure(figsize=(10, 6))
    if var == 'prediction_error':
        label = 'Prediction Error'
        calculation = 'Model Score - Human Score'
        xlims = [-4, 4]
    elif var == 'absolute_prediction_error':
        label = 'Absolute prediction Error'
        calculation = '|Model Score - Human Score|'
        xlims = [0, 5]
    else: 
        continue
    
    sns.kdeplot(df_discrepancy[var], label=label, fill=True, alpha=0.5)
    
    if var == 'prediction_error':
        plt.axvline(x=0, color='gray', linestyle='dashed', alpha=0.3, label='Optimum (No Dicrepancy)') 
        
    plt.axvline(x=mean_error, color='red', linestyle='dashed', alpha=0.6, label='Mean Prediction Error')
    plt.xlabel('Discrepancy')
    plt.ylabel('Density')
    plt.title(f'Discrepancy between Model Score and Human Score on Mohler Dataset\n({calculation})')
    plt.legend()
    plt.xlim(xlims)
    plt.show()


# +
# QS 
percentile_cols = [col for col in df_discrepancy.columns if 'percentile_' in col]
percentile_cols = [col for col in percentile_cols if not col == "percentile_rank_grouped"]
df_bar_plot = df_discrepancy.filter(
    ['human_score'] + percentile_cols
).drop_duplicates().sort_values(by=['human_score']).reset_index(drop=True)

df_bar_plot

# +
#x-Axis
x = np.arange(len(df_bar_plot['human_score'])) 
width = 0.2

# Farben für die Perzentile
colors = cm.Blues(np.linspace(0.5, 0.9, len(percentile_cols)))

# Figur und Achse erstellen
fig, ax = plt.subplots(figsize=(10, 6))

# Balken für jedes Perzentil hinzufügen
for i, col in enumerate(percentile_cols):
    legend_label = f"{col.split('_')[1]}%-Percentile"  
    ax.bar(
        x + i * width, 
        df_bar_plot[col],
        width,
        label=legend_label,
        color=colors[i]
    )

# Achsen und Titel anpassen
ax.set_xticks(x + width * (len(percentile_cols) - 1) / 2)
ax.set_xticklabels(df_bar_plot['human_score'])
ax.set_xlabel('Human Score')
ax.set_ylabel('Absolute Prediction Error')
ax.set_title('Grouped Barplot of Percentiles by Human Score')
ax.legend(title='')

# Layout und Anzeige
plt.tight_layout()
plt.show()
# -

customized_html_export(
    df_discrepancy.query("absolute_prediction_error > percentile_75").drop(columns=percentile_cols).sort_values(by='percentile_rank_grouped', ascending=False),
    filename = './exports/mohler_high_discrepency.html',
    display_simple_table_here=False
)

a = df_discrepancy.query("model_score < human_score").shape[0]
b = df_discrepancy.query("model_score > human_score").shape[0]
print(f"model_score < human_score: {a} cases.")
print(f"model_score > human_score:  {b} cases.")


df_error_per_ques.head(20)

df_error_per_ques.tail()

# <h4 style="
#     font-family: 'Times New Roman', Times, serif;
#     font-size: 18px;
#     color: #2C3E50;
#     text-align: left;
#     border-bottom: 1px dotted #BDC3C7;
#     padding-bottom: 3px;
#     margin-top: 18px;
#     margin-bottom: 8px;
# ">
#     OS Dataset
# </h4>
#
# OS verfügt über ein quasiintervallskaliertes True-Label. Das bedeutet:
#
# * Regressionsmaße a la $R^2$, RMSE, MAE oder Pearsons $r$ sinnvoll mit gewissen Einschränkungen, da nur quasi-intervallskalierte True-Labels.
# * Weighted Cohens Kappa zwischen ordinal transformierter Vorhersage und True-Label sinnig.
# * ROC-AUC-Werte in meinen Augen nicht sinnvoll berechenbar.

# __Über alle Fragen hinweg__

# +
qs = []

# specify alias
for alias in ['os_en', 'os_de']:
    print(f"\nAlias: {alias}")
    # initialize results dictionairy
    formal_results[alias] = {}
    
    # copy dataset
    df = evaluation_data.datasets[alias].df.copy()
    
    # specify and prepare model and human_scores     
    # multiplication with 2 because Cohen Kappa needs integer values
    # and is not suitable with decimals
    model_scores = df['model_score_rounded'].astype(int)
    human_scores = df['human_score'].astype(int)
    
    # qs to check that bevor using int() model_score consists out of natural numbers
    qs.append({
        'model_score_rounded': np.sort((df['model_score_rounded']).unique()),
        'human_score': np.sort((df['human_score']).unique())
    })

    # Calculate Kappa Scores
    formal_results[alias]['kappa'] = calculate_kappa_scores(
        human_scores, model_scores
    )

print('\n')
for el in qs: print(el)

# -

# __Für jede Frage separat__

# +
results_df = pd.DataFrame(
    columns=[
        "alias",
        "question_id_2",
        "kappa_unweighted",
        "kappa_linear",
        "kappa_quadratic"
    ]
)

for alias in ["os_en", "os_de"]:
    # Retrieve all unique question_ids for the current alias
    question_ids = np.sort(evaluation_data.datasets[alias].df["question_id_2"].unique())
    
    for cur_id in question_ids:
        # Filter the DataFrame for the current question_id_2
        df_subset = evaluation_data.datasets[alias].df.query(f"question_id_2 == {cur_id}").copy()
        
        # Prepare scores as integer values
        model_scores = df_subset["model_score_rounded"].astype(int)
        human_scores = df_subset["human_score"].astype(int)
        
        # Calculate Kappa metrics
        kappa_result = calculate_kappa_scores(human_scores, model_scores, shut_up=True)
        
        # Append
        results_df.loc[len(results_df)] = {
            "alias": alias,
            "question_id_2": cur_id,
            "kappa_unweighted": kappa_result["unweighted"],
            "kappa_linear": kappa_result["linear"],
            "kappa_quadratic": kappa_result["quadratic"]
        }
# -

results_df.query("alias == 'os_en'").sort_values(by='kappa_quadratic', ascending=False)

results_df.query("alias == 'os_de'").sort_values(by='kappa_quadratic', ascending=False)

# +
special_results = pd.concat([
    special_results_os_dataset['os_en'],
    special_results_os_dataset['os_de']
])

special_results['question_id_2'] = special_results['filter'].str[-1].astype(int)

combined_results_for_paper_OS = pd.merge(
    results_df,
    special_results,
    on=['alias', 'question_id_2']
).filter(
    ['Detailed Name', 'question_id_2', 'kappa_quadratic', 'Pearson']
).sort_values(by = ['Detailed Name', 'kappa_quadratic'], ascending = False)

combined_results_for_paper_OS


# -
pd.merge(
    combined_results_for_paper_OS,
    pd.concat([dimtables['os_en'], dimtables['os_de']]).filter(['question_id_2', 'custom_question_id']).drop_duplicates(),
    on = 'question_id_2'
)[6:13].sort_values(by='kappa_quadratic', ascending=False)

# <h4 style="
#     font-family: 'Times New Roman', Times, serif;
#     font-size: 18px;
#     color: #2C3E50;
#     text-align: left;
#     border-bottom: 1px dotted #BDC3C7;
#     padding-bottom: 3px;
#     margin-top: 18px;
#     margin-bottom: 8px;
# ">
#     SemEval
# </h4>
#
# SemEval verfügt über ein binäres True-Label. Das bedeutet:
# * Regressionsmaße a la $R^2$, RMSE, MAE oder Pearsons $r$ in meinen Augen völliger Quatsch.
# * ROC-AUC wunderbar berechenbar.
# * Alle auf Basis der Konfusionsmatrix gebildeten Metriken (F1, Acccuracy etc.) bieten sich an. Aber: gewählter Schwellenwert maßgebend.
# * Optimierung des Schwellenwerts lässt Maximum aus Accuracy, F1 etc rausholen

# +
# specify alias
alias = 'semeval'

# initialize results dictionairy
formal_results[alias] = {}

# # copy dataset
df = evaluation_data.datasets[alias].df.copy()

# build dummies (0/1-coded)
human_scores = df['metadata_original_label'].apply(lambda x: 1 if x == 'correct' else 0)
model_scores = df['model_score_original'].apply(lambda x: 1 if x >= 50 else 0)

# extract original score
model_score_original = df['model_score_original']
# -

# __Cohens Kappa__

# calculate Kappa Scores
formal_results[alias]['kappa'] = calculate_kappa_scores(
    human_scores, model_scores
)

# __Classfication metrics__ (using threshold in the middle $\rightarrow$ no optimization)

# calculate Accuracy, F1 etc when using exactly the middle as threshold (no optimization of threshold)
formal_results[alias]['classification_metrics'] = evaluate_classification(
    y_true = human_scores,
    y_pred = model_scores
)

# __ROC-AUC and ROC-Curve__

# +
# Calculate ROC-AUC (schwellenwertunabhängig)
formal_results[alias]['ROC_AUC'] = roc_auc_score(
    human_scores,
    model_score_original
)

print(f"ROC-AUC: {formal_results[alias]['ROC_AUC']}")
# -

# Plot ROC-Curve
fpr, tpr, thresholds = roc_curve(human_scores, model_score_original)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {formal_results[alias]['ROC_AUC']:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('')#(f'ROC Curve for Dataset: {alias}')
plt.legend(loc='best')
plt.grid()
filename = 'SemEval_roc_curve'
plt.savefig(f"./exports/{filename}.png")
tikzplotlib.save(f"./exports/{filename}.tex")
plt.show()

# __Threshold Optimization__

# +
# Optimization
results, f1_values = optimize_threshold(
    y_true_binary = human_scores,
    y_pred_score = model_score_original,
    thresholds = thresholds,
    optimize_for="f1"
)

# 3) Ergebnisse anzeigen
#print("Optimized Metrics:", results)

# Plot results
plot_metric_vs_threshold(
    thresholds,
    f1_values,
    results["best_threshold"],
    metric_name="F1",
    file_path_prefix = './exports/threshold_optimization_F1_semeval'
)


# +
# Optimization
results, acc_values = optimize_threshold(
    y_true_binary = human_scores,
    y_pred_score = model_score_original,
    thresholds = thresholds,
    optimize_for="accuracy"
)

# 3) Ergebnisse anzeigen
#print("Optimized Metrics:", results)

# Plot results
plot_metric_vs_threshold(
    thresholds,
    acc_values,
    results["best_threshold"],
    metric_name="Accuracy",
    file_path_prefix = './exports/threshold_optimization_Acc_semeval'
)

# -

# <h4 style="
#     font-family: 'Times New Roman', Times, serif;
#     font-size: 18px;
#     color: #2C3E50;
#     text-align: left;
#     border-bottom: 1px dotted #BDC3C7;
#     padding-bottom: 3px;
#     margin-top: 18px;
#     margin-bottom: 8px;
# ">
#     SemEval (calculations separately for corpi 'beetle' and 'sciEntsBank')
# </h4>

# +
# Corpus
for corpus in ['beetle', 'sciEntsBank']:
   
    # specify alias
    alias = 'semeval'
    
    # fetch dataset
    df = evaluation_data.datasets[alias].df.copy()
    df = df.query(f"metadata_corpus == '{corpus}'")
    n_rows = df.shape[0]
    
    # re-specify alias
    alias = f"{alias}_{corpus}"
    
    print("===========================================================")
    print(f"dataset: {alias}\nnumber of Rows: {n_rows}")
    print("===========================================================")
    
    # initialize results dictionairy
    formal_results[f"{alias}"] = {}
    
    # build dummies (0/1-coded)
    human_scores = df['metadata_original_label'].apply(lambda x: 1 if x == 'correct' else 0)
    model_scores = df['model_score_original'].apply(lambda x: 1 if x >= 50 else 0)
    
    # extract original score
    model_score_original = df['model_score_original']
    
    
    # calculate Kappa Scores
    formal_results[alias]['kappa'] = calculate_kappa_scores(
        human_scores, model_scores
    )
    
    # calculate Accuracy, F1 etc when using exactly the middle as threshold (no optimization of threshold)
    formal_results[alias]['classification_metrics'] = evaluate_classification(
        y_true = human_scores,
        y_pred = model_scores
    )
    
    # Calculate ROC-AUC (schwellenwertunabhängig)
    formal_results[alias]['ROC_AUC'] = roc_auc_score(
        human_scores,
        model_score_original
    )
    
    print(f"ROC-AUC: {formal_results[alias]['ROC_AUC']}")
    
    
    # Plot ROC-Curve
    fpr, tpr, thresholds = roc_curve(human_scores, model_score_original)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {formal_results[alias]['ROC_AUC']:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curve for Dataset: {alias}')
    plt.legend(loc='best')
    plt.grid()
    filename = f'semeval_roc_curve_{alias}'
    plt.savefig(f"./exports/{filename}.png")
    tikzplotlib.save(f"./exports/{filename}.tex")
    plt.show()
    
    # Optimization F1
    results, f1_values = optimize_threshold(
        y_true_binary = human_scores,
        y_pred_score = model_score_original,
        thresholds = thresholds,
        optimize_for="f1"
    )
    
    plot_metric_vs_threshold(
        thresholds,
        f1_values,
        results["best_threshold"],
        metric_name="F1",
        file_path_prefix = f'./exports/threshold_optimization_F1_{alias}'
    )
    
    # Optimization Acc
    results, acc_values = optimize_threshold(
        y_true_binary = human_scores,
        y_pred_score = model_score_original,
        thresholds = thresholds,
        optimize_for="accuracy"
    )
    
    plot_metric_vs_threshold(
        thresholds,
        acc_values,
        results["best_threshold"],
        metric_name="Accuracy",
        file_path_prefix = f'./exports/threshold_optimization_Acc_{alias}'
    )
    
    

# -
with open("formal_results.json", "w") as file:
    json.dump(formal_results, file, indent=4) 
