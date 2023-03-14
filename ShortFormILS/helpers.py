import os
import random
from matplotlib import pyplot as plt

import numpy as np


def seed_everything(seed=42):
    """
    Seed everything.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def round_to_class(y_bf):
    """Round to class."""
    y_bf = np.where(y_bf == -5, 0, y_bf)
    y_bf = np.where(y_bf == 1, 2, y_bf)
    y_bf = np.where(y_bf == -1, 2, y_bf)
    y_bf = np.where(y_bf == -3, 1, y_bf)
    y_bf = np.where(y_bf == 3, 3, y_bf)
    y_bf = np.where(y_bf == 5, 4, y_bf)
    return y_bf


def round_to_value(y_bf):
    """Round to value classes."""
    y_bf = np.where(y_bf == -3, -1, y_bf)
    y_bf = np.where(y_bf == -7, -3, y_bf)
    y_bf = np.where(y_bf == -5, -3, y_bf)
    y_bf = np.where(y_bf == -11, -5, y_bf)
    y_bf = np.where(y_bf == -9, -5, y_bf)
    y_bf = np.where(y_bf == 3, 1, y_bf)
    y_bf = np.where(y_bf == 7, 3, y_bf)
    y_bf = np.where(y_bf == 5, 3, y_bf)
    y_bf = np.where(y_bf == 11, 5, y_bf)
    y_bf = np.where(y_bf == 9, 5, y_bf)
    return y_bf


def round_to_dim(y_bf):
    """Round to dim classes."""
    y_bf = np.where(y_bf <= 1, 0, y_bf)
    y_bf = np.where(y_bf == 2, 1, y_bf)
    y_bf = np.where(y_bf >= 3, 2, y_bf)
    return y_bf


def round_to_critical_cases_classes(y_bf):
    """Round to critical cases classes."""
    y_bf = np.where(y_bf == 1, 0.5, y_bf)
    y_bf = np.where(y_bf == 2, 1.5, y_bf)
    y_bf = np.where(y_bf == 3, 2.5, y_bf)
    y_bf = np.where(y_bf == 4, 3, y_bf)
    return y_bf


def plot_result(metric_dict, regressorname, filename):
    """Plot the results."""
    global plot_counter
    plot_counter += 1
    plt.figure(plot_counter)
    # sort based on avg_score
    # only take subset of size n with highest accuracy
    k_feat = sorted(
        metric_dict.keys(), key=lambda k: metric_dict[k]["avg_score"], reverse=True
    )[:10]
    # sort again based on index
    k_feat = sorted(k_feat)
    avg = [metric_dict[k]["avg_score"] for k in k_feat]

    upper, lower = [], []
    for k in k_feat:
        upper.append(metric_dict[k]["avg_score"] + metric_dict[k]["std_dev"])
        lower.append(metric_dict[k]["avg_score"] - metric_dict[k]["std_dev"])

    plt.fill_between(k_feat, upper, lower, alpha=0.2, color="blue", lw=1)

    plt.plot(k_feat, avg, color="blue", marker="o", markersize=3)
    plt.ylabel("Accuracy +/- Standard Deviation")
    plt.xlabel("Best subset (k)")
    feature_min = len(metric_dict[k_feat[0]]["feature_idx"])
    feature_max = len(metric_dict[k_feat[-1]]["feature_idx"])
    plt.title(
        "Exhaustive Feature Selection (min {} features, max {} features)".format(
            feature_min, feature_max
        )
    )
    plt.xticks(
        k_feat, [str(metric_dict[k]["feature_idx"]) for k in k_feat], rotation=70
    )
    # plot zoom out
    plt.ylim([min(lower) - 0.1, max(upper) + 0.1])
    plt.subplots_adjust(bottom=0.3)
    # create folder if not exists
    if not os.path.exists("plots"):
        os.makedirs("plots")
    # Save the plot
    plt.savefig("plots/{}_efs_{}_.png".format(filename, regressorname), dpi=300)
    plt.close(plot_counter)
