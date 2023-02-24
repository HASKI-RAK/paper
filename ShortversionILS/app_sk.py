import os
from matplotlib import pyplot as plt
import pandas as pd
from pprint import pprint
from sklearn import linear_model
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, f_regression, mutual_info_regression
from sklearn import neural_network
from sklearn.metrics import f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin, RegressorMixin
from helpers import seed_everything, round_to_class, round_to_value, round_to_dim
import numpy as np

seed_everything(1337)
plot_counter = 0


def plot_result(metric_dict, regressorname, filename):
    # new plot
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

# We need a custom scorer in order to compensate the relation between 11 features summed and 5 features summed giving y
def custom_scorer(estimator: ClassifierMixin | RegressorMixin, x, _y, granularity=1):
    y = round_to_value(_y)
    if granularity == 1:
        y = round_to_dim(round_to_class(y))
    if granularity == 2:
        y = round_to_class(y)
    y_pred = estimator.predict(x)
    acc = float((y == y_pred).sum()) / y_pred.shape[0]
    score = estimator.score(x, y) # r2 for regressors and accuracy for classifiers
    return acc, score

def exhaustive_stepwise_regression(
    _x, _y, model, filename, cv=2, scoring="neg_root_mean_squared_error", granularity=1
):
    acc = -1
    score = -1
    def scorer_wrapper(estimator, x, y):
        _acc, _score = custom_scorer(estimator, x, y, granularity)
        acc = _acc
        score = _score
        return _acc

    efs = EFS(
        model,
        min_features=5,
        max_features=5,
        scoring=scorer_wrapper,
        print_progress=True,
        cv=cv,
        n_jobs=-1,
    )

    efs = efs.fit(_x, _y)
    # create folder if not exists
    if not os.path.exists("plots"):
        os.makedirs("plots")
    with open(
        "plots/{}_efs_{}_cv{}.log".format(
            filename, efs.estimator.__class__.__name__, cv
        ),
        "w",
    ) as log_file:
        pprint("####################")
        pprint(efs.best_feature_names_, stream=log_file)


        # Select columns in X based on the best feature names:
        x_bestfeatures = _x[np.array(efs.best_feature_names_)]
        # Sum up the values of selected features to get new y:
        y_bf = x_bestfeatures.sum(axis=1)
        if granularity == 1:
            y_bf = round_to_dim(round_to_class(round_to_value(y_bf)))
        if granularity == 2:
            assert(float((round_to_class(_y) == round_to_class(y_bf)).sum()) / y_bf.shape[0])
            y_bf = round_to_class(round_to_value(y_bf))

        if granularity == 3:
            # Map -11,-9 to -5 and 9,11 to 5, etc.:
            y_bf = round_to_value(y_bf)
            # y_bf = y_bf * 2
            # y_bf_more = np.where(y_bf < 0, y_bf-1, y_bf+1)
            # y_bf_less = np.where(y_bf < 0, y_bf+1, y_bf-1)
            # isthesame_bool = np.isin(y_bf, y_bf_more)
        acc = float((_y == y_bf).sum()) / y_bf.shape[0]
        pprint("Test set accuracy: %.2f %%" % (acc * 100), stream=log_file)
        # print f1 score
        # pprint(f1_score(_y, y_pred, average='macro', zero_division=0))

        # print model name and settings
        pprint("Model name: %s" % efs.estimator.__class__.__name__, stream=log_file)
        pprint("Model settings: %s" % efs.estimator.get_params(), stream=log_file)
        pprint(
            "Best accuracy score (higher is better): %.4f" % efs.best_score_,
            stream=log_file,
        )
        metric_dict = efs.get_metric_dict()
        best_feature_idx = [
            k
            for k in metric_dict.keys()
            if metric_dict[k]["feature_idx"] == efs.best_idx_
        ][0]
        test = efs.get_metric_dict()[best_feature_idx]["std_dev"]
        pprint("Standard deviation of the best score: %.4f" % test, stream=log_file)
        pprint("Best subset (indices): {}".format(efs.best_idx_), stream=log_file)
        pprint(
            "Best subset (corresponding names): {}".format(efs.best_feature_names_),
            stream=log_file,
        )

    metric_dict = efs.get_metric_dict()

    plot_result(metric_dict, efs.estimator.__class__.__name__, filename)
    return [
        filename,
        efs.best_score_,
        acc,
        efs.best_idx_,
        efs.best_feature_names_,
        efs.estimator.__class__.__name__,
        scoring
    ]


def recursive_feature_elimination(
    _x,
    _y,
    model,
    filename,
    cv=2,
    scoring="neg_root_mean_squared_error",
    min_features_to_select=5,
):
    rfecv = RFECV(
        estimator=model,
        step=1,
        cv=cv,
        scoring=scoring,
        min_features_to_select=min_features_to_select,
        n_jobs=-1,
    )
    rfecv.fit(_x, _y)
    # create folder if not exists
    if not os.path.exists("plots"):
        os.makedirs("plots")
    with open(
        "plots/{}_rfe_{}_cv{}.log".format(
            filename, rfecv.estimator.__class__.__name__, cv
        ),
        "w",
    ) as log_file:
        pprint("####################")
        pprint(rfecv.support_, stream=log_file)
        pprint(rfecv.ranking_, stream=log_file)

        # print model name and settings
        pprint("Model name: %s" % rfecv.estimator.__class__.__name__, stream=log_file)
        pprint("Model settings: %s" % rfecv.estimator.get_params(), stream=log_file)
        pprint(
            "Optimal number of features: {}".format(rfecv.n_features_), stream=log_file
        )
        pprint(
            "Mean test score: {}".format(rfecv.cv_results_["mean_test_score"]),
            stream=log_file,
        )
        pprint(
            "Mean std score: {}".format(rfecv.cv_results_["std_test_score"]),
            stream=log_file,
        )
        pprint("Best subset (indices): {}".format(rfecv.support_), stream=log_file)
        pprint(
            "Best subset (corresponding names): {}".format(_x.columns[rfecv.support_]),
            stream=log_file,
        )
    # plot_result(rfecv.grid_scores_, rfecv.estimator.__class__.__name__, filename)

    return [
        filename,
        rfecv.cv_results_,
        rfecv.support_,
        _x.columns[rfecv.support_],
        rfecv.estimator.__class__.__name__,
    ]


# def nullhypothesis_test(_x, _y, model, filename, cv=2, scoring='neg_root_mean_squared_error'):
#     # every combination of 5 features in X has the same accuracy in predicting y
#     # if the accuracy is the same, the null hypothesis is true
#     for i in range(5, _x.shape[1]+1):
#         # create folder if not exists
#         if not os.path.exists('plots'):
#             os.makedirs('plots')
#         with open("plots/{}_nullhypothesis_{}_cv{}.log".format(filename, model.__class__.__name__, cv), "w") as log_file:
#             pprint("####################")
#             pprint("Testing null hypothesis for {} features".format(i),stream=log_file)
#             # print model name and settings
#             pprint('Model name: %s' % model.__class__.__name__,stream=log_file)
#             pprint('Model settings: %s' % model.get_params(),stream=log_file)
#             pprint('Testing null hypothesis for {} features'.format(i),stream=log_file)
#             # every combination of i features in X has the same accuracy in predicting y
#             # if the accuracy is the same, the null hypothesis is true
#             for combination in combinations(_x.columns, i):
#                 # print combination
#                 pprint('Testing combination: {}'.format(combination),stream=log_file)
#                 # fit model
#                 model.fit(_x[list(combination)], _y)
#                 # predict
#                 y_pred = model.predict(_x[list(combination)])
#                 # print accuracy
#                 acc = float((_y == y_pred).sum()) / y_pred.shape[0]
#                 pprint('Test set accuracy: %.2f %%' % (acc*100),stream=log_file)
#                 # print f1 score
#                 #pprint(f1_score(_y, y_pred, average='macro', zero_division=0),stream=log_file)


#### MAIN ####
# Load data
def load_data(name="D1SMRC", granularity=1):
    filename = "./{}.xlsx".format(name)
    df = pd.read_excel(filename, sheet_name="HASKI Pretest (ILS, LIST-K & BF")

    # Select every column except the last one
    X = df.iloc[:, :-3]
    # n sample size of the data
    n = X.shape[0]
    # Select the last column
    y = df.iloc[:, -granularity]
    return X, y, n


# Regression models
models = [
    (linear_model.LinearRegression(positive=True), "neg_root_mean_squared_error"),
    (linear_model.ElasticNet(), "neg_mean_squared_error"),
    (linear_model.Ridge(), "neg_mean_squared_error"),
    (linear_model.Lasso(), "neg_mean_squared_error"),
    (linear_model.Perceptron(), "neg_mean_squared_error"),
    (
        neural_network.MLPRegressor(random_state=1337, hidden_layer_sizes=(100), max_iter=1500),
        "neg_mean_squared_error",
    ),
    (DecisionTreeClassifier(), "accuracy"),
    (RandomForestClassifier(), "accuracy"),
    (linear_model.PassiveAggressiveClassifier(), "accuracy"),
    (linear_model.RidgeClassifier(), "accuracy"),
    (linear_model.SGDClassifier(), "accuracy"),
]

datasets = ["D1SMRC", "D2SMRC", "D3SMRC", "D4SMRC"]
results = []


# ### Stradegy 1: Exhaustive Feature Selection
def esr(
    exhaustive_stepwise_regression,
    load_data,
    models,
    datasets,
    results,
    granularity=1,
    default_scoring=None,
):
    pprint("####################")
    pprint("Exhaustive Feature Selection")
    pprint("Granularity: {}".format(granularity))
    pprint("Default scoring: {}".format(default_scoring))
    pprint("####################")
    for dataset_name in datasets:
        X, y, _ = load_data(dataset_name, granularity=granularity)
        for model, scoring in models:
            scoring = default_scoring if default_scoring else scoring
            results.append(exhaustive_stepwise_regression(
                    X, y, model, dataset_name, scoring=scoring, granularity=granularity
                )
            )

    df = pd.DataFrame(
        results,
        columns=[
            "Dataset",
            "Score (higher is better)",
            "Accuracy (higher is better)"
            "Best subset (indices)",
            "Best subset (corresponding names)",
            "Model",
            "Scoring"
        ],
    )
    df.to_excel("plots/efs_{}_{}_results.xlsx".format(granularity,default_scoring), index=False)


# ### Stradegy 2: Recursive Feature Elimination
def rfe(recursive_feature_elimination, load_data, models, datasets):
    results = []
    for dataset_name in datasets:
        X, y, _ = load_data(dataset_name)
        for model, scoring in models:
            results.append(
                recursive_feature_elimination(
                    X, y, model, dataset_name, scoring=scoring
                )
            )
    # Logistic regression with cv=0 is equivalent to exhaustive search
    # results.append(recursive_feature_elimination(X, y, linear_model.LogisticRegression(solver='lbfgs', max_iter=100), dataset_name, cv=0))

    df = pd.DataFrame(
        results,
        columns=[
            "Dataset",
            "Score",
            "Best subset (indices)",
            "Best subset (corresponding names)",
            "Model",
        ],
    )
    df.to_csv("plots/rfe_results.csv", index=False)


# ### Stradegy 3: Principal Component Analysis
def pca(load_data, datasets):
    results = []
    for dataset_name in datasets:
        X, y, _ = load_data(dataset_name)
        pca = PCA(n_components=5)
        X = pca.fit_transform(X, y)
        results.append(
            [
                dataset_name,
                pca.explained_variance_ratio_,
                pca.singular_values_,
                pca.components_,
            ]
        )
        df_components = pd.DataFrame(pca.components_)
        df_components.to_excel(
            "plots/pca_{}_components.xlsx".format(dataset_name), index=False
        )
    df = pd.DataFrame(
        results,
        columns=[
            "Dataset",
            "Explained variance ratio",
            "Singular values",
            "Components",
        ],
    )
    df.to_csv("plots/pca_results.csv", index=False)


def ftest(load_data, datasets):
    results = []
    for dataset_name in datasets:
        X, y, _ = load_data(dataset_name)
        # X shape:
        pprint(X.shape)
        f_test, _ = f_regression(X, y)
        f_test /= np.max(f_test)
        # mutual information
        mi = mutual_info_regression(X, y)
        mi /= np.max(mi)
        # plot the results
        plt.matshow(np.c_[f_test, mi], cmap=plt.cm.Blues, vmin=0, vmax=1)
        plt.yticks(range(X.shape[1]), range(X.shape[1]))
        plt.xticks([0, 1], ["F-test", "Mutual Information"])
        plt.colorbar()
        # plt.figure(figsize=(15, 5))
        # for i in range(10):
        #     plt.subplot(1, 11, i + 1)
        #     plt.tight_layout()
        #     plt.scatter(X.iloc[:, i+1], y, s=10)
        #     plt.xlabel("$x_{}$".format(i + 1), fontsize=14)
        #     plt.ylabel("F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]), fontsize=16)
        plt.savefig("plots/ftest_{}.png".format(dataset_name))
        results.append([dataset_name, f_test])
    df = pd.DataFrame(results, columns=["Dataset", "F-test"])
    df.to_csv("plots/ftest_results.csv", index=False)


if __name__ == "__main__":
    plt.switch_backend("agg")
    # Exhaustive feature selection with 3 different granularities.
        # 2 Means trying to guess the fine grained dimension: strongly aktive, moderately aktive, balanced, moderately reflective, strongly reflective
    esr(
        exhaustive_stepwise_regression,
        load_data,
        models,
        datasets,
        results,
        granularity=2,
        default_scoring="r2",
    )
    results = []
        # 3 Means trying to guess the exact score: -11, ... ,0,... 11
    esr(
        exhaustive_stepwise_regression,
        load_data,
        models,
        datasets,
        results,
        granularity=3,
        default_scoring="r2",
    )
    results = []
    # 1 Means trying to guess the dimension: aktive, balanced, reflective
    esr(
        exhaustive_stepwise_regression,
        load_data,
        models,
        datasets,
        results,
        granularity=1,
        default_scoring="r2",
    )


