# Data Manipulation Libraries
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

# Visualization Libraries
import matplotlib

# Non-interactive backend so that file saving doesn't consume too much memory (no need for mlp GUI)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
# from pysankey import sankey


# Machine Learning Libraries
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# SHAP scores
from shap import summary_plot, Explainer
from shap.maskers import Independent

# Miscellaneous System libraries
from datetime import datetime
import os
from pathlib import Path
from typing import Any

# Custom Utilities
from utils.stdout import StdOut


def with_tuning(classifier, rand_state, nsplits: int, parameters: dict[str, Any]):
    return GridSearchCV(
        classifier,
        param_grid=parameters,
        cv=KFold(n_splits=nsplits, shuffle=True, random_state=rand_state),
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )


def rf_classifier(x: DataFrame, y: Series, rand_state: int, tuning: bool, nsplits: int, stdout: StdOut):
    stdout.write("Random Forest")
    clf = RandomForestClassifier(random_state=rand_state)
    if tuning:
        stdout.write("Tuning Hyperparameters")
        clf = with_tuning(
            classifier=clf,
            rand_state=rand_state,
            nsplits=nsplits,
            parameters={
                "max_features": np.arange(1, x.shape[1] + 1),
                "min_samples_split": np.arange(2, 11),
                "min_samples_leaf": np.concatenate([np.arange(1, 11), [100, 150]]),
            },
        )

    result = clf.fit(x, y)
    return result.best_estimator_ if tuning else result


def svm_classifier(x: DataFrame, y: Series, rand_state: int, tuning: bool, nsplits: int, stdout: StdOut) -> BaseEstimator:
    stdout.write("Support Vector Machine")
    clf = SVC(random_state=rand_state, kernel="linear", probability=True)
    if tuning:
        stdout.write("Tuning Hyperparameters")
        clf = with_tuning(
            classifier=clf,
            rand_state=rand_state,
            nsplits=nsplits,
            parameters={
                "kernel": ["linear"],
                "gamma": [0.001, 0.01, 0.1, 1, 10],
                "C": [0.01, 0.1, 1, 10, 50, 100, 500, 1000, 5000, 10000],
            },
        )

    result = clf.fit(x, y)
    return result.best_estimator_ if tuning else result


def xgb_classifier(x: DataFrame, y: Series, rand_state: int, tuning: bool, nsplits: int, stdout: StdOut):
    stdout.write("XGBoost")
    clf = XGBClassifier(random_state=rand_state, objective="binary:logistic")
    if tuning:
        stdout.write("Tuning Hyperparameters")
        clf = with_tuning(
            classifier=clf,
            rand_state=rand_state,
            nsplits=nsplits,
            parameters={
                "n_estimators": [int(x) for x in np.linspace(100, 500, 5)],
                "max_depth": [int(x) for x in np.linspace(3, 9, 4)],
                "gamma": [0.01, 0.1],
                "learning_rate": [0.001, 0.01, 0.1, 1],
            },
        )

    result = clf.fit(x, y)
    return result.best_estimator_ if tuning else result


def knn_classifier(x: DataFrame, y: Series, rand_state: int, tuning: bool, nsplits: int, stdout: StdOut):
    stdout.write("K-Nearest Neighbors")
    clf = KNeighborsClassifier()
    if tuning:
        stdout.write("Tuning Hyperparameters")
        clf = with_tuning(
            classifier=clf,
            rand_state=rand_state,
            nsplits=nsplits,
            parameters={
                "leaf_size": list(range(1, 50)),
                "n_neighbors": list(range(1, min(len(x) - 1, 30) + 1)),
                "p": [1, 2],
            },
        )

    result = clf.fit(x, y)
    return result.best_estimator_ if tuning else result


def mlp_classifier(x: DataFrame, y: Series, rand_state: int, tuning: bool, nsplits: int, stdout: StdOut):
    stdout.write("Multi-Layer Perceptron")
    clf = MLPClassifier(random_state=rand_state, max_iter=2000)
    if tuning:
        stdout.write("Tuning Hyperparameters")
        clf = with_tuning(
            classifier=clf,
            rand_state=rand_state,
            nsplits=nsplits,
            parameters={
                "hidden_layer_sizes": [
                    (50, 50, 50),
                    (50, 100, 50),
                    (100,),
                    (100, 100),
                    (100, 100, 100),
                ],
                "activation": ["tanh", "relu"],
                "solver": ["sgd", "adam"],
                "alpha": [0.0001, 0.001, 0.01, 0.05, 0.1],
                "learning_rate": ["constant", "adaptive"],
                "learning_rate_init": [0.001, 0.01, 0.1],
                "momentum": [0.1, 0.2, 0.5, 0.9],
            },
        )

    result = clf.fit(x, y)
    return result.best_estimator_ if tuning else result


def voting_classifier(
    x: DataFrame,
    y: DataFrame,
    voting: str,
    names: list[str],
    classifiers: list[Any],
    stdout: StdOut,
):
    stdout.write("Voting Classifier")
    return VotingClassifier(estimators=list(zip(names, classifiers)), voting=voting).fit(x, y)


def standard_scalar(x: DataFrame) -> DataFrame:
    return pd.DataFrame(StandardScaler().fit_transform(x), columns=x.columns)


def expression_direction(*scores: list[float]):
    positive_count = sum(1 for score in scores if score > 0)
    negative_count = sum(1 for score in scores if score < 0)

    if positive_count > negative_count:
        return "Overexpressed"
    elif negative_count > positive_count:
        return "Underexpressed"
    else:
        return "Inconclusive"


def classify_features(
    input_df: DataFrame,
    # None = use all features
    selected_features: list[str] | None,
    stdout: StdOut,
    rand_state: int,
    test_size: float,
    use_normalization: bool,
    use_tuning: bool,
    nsplits: int,
    use_rf: bool,
    use_svm: bool,
    use_xgb: bool,
    use_knn: bool,
    use_mlp: bool,
    voting_type: str,  # soft or hard
    use_visualizations: bool,
    use_igenes: bool,
    output_dir: str,
    stem: str,
):
    id_column = "ID"
    y_label_col = "Type"

    for c in [id_column, y_label_col]:
        if c not in input_df.columns:
            stdout.write(f"Invalid format: Missing column '{c}' in CIGT file.")
            return

    parsed_input_df = input_df.drop(columns=[id_column])

    X = parsed_input_df.drop(columns=[y_label_col])
    if selected_features is None:
        stdout.write("No selected features provided. Using all features present")
    elif len(selected_features) == 0:
        stdout.write("Provided empty list of features. Exiting...")
        return
    else:
        # Keep selected features only
        X = X[X.columns[X.columns.isin(selected_features)]]
    Y = parsed_input_df[y_label_col]

    parsed_input_df = X.join(Y)  # ignore unselected columns
    melted_df = parsed_input_df.melt(id_vars=y_label_col, var_name="Feature", value_name="Value")
    melted_path = os.path.join(output_dir, f"{stem}_Collapsed-Feature-Values.csv")
    melted_df.to_csv(melted_path, index=False)
    stdout.write(f"Saved collapsed feature values to {melted_path}")

    # Calculating correlation matric
    correlation_df = X.corr()
    correlation_path = os.path.join(output_dir, f"{stem}_Feature-Correlations.csv")
    correlation_df.to_csv(correlation_path, index=False)
    stdout.write(f"Saved inter-feature correlations to {correlation_path}")

    stdout.write("Feature Classification")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x: DataFrame
    x_t: DataFrame
    y: Series
    y_t: Series
    x, x_t, y, y_t = train_test_split(X, Y, test_size=test_size, random_state=rand_state)

    if use_normalization:
        stdout.write("Normalizing DataFrame")
        x = standard_scalar(x)

    names: list[str] = []
    classifiers: list[BaseEstimator] = []
    model_shaps: list[np.ndarray] = []
    explainers: list[Explainer] = []
    model_confusion: list[DataFrame] = []
    feature_hhi_weights = []  # weights of each feature *per model*
    normalized_features_importances = []  # importances for each feature *per model*
    roc_scores: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # fpr, tpr, thresholds

    # Kernel Explainer caused a strange bug with PyInstaller. Namely, it caused a residual
    # window to popup after calling shap_values(). Letting the shap library decide which
    # window to use (e.g ExactExplainer) is better.
    if use_rf:
        names.append("Random Forest")
        rf = rf_classifier(x, y, rand_state, use_tuning, nsplits, stdout=stdout)
        classifiers.append(rf)
        explainers.append(Explainer(rf))
    if use_svm:
        names.append("Support Vector Machine")
        svm = svm_classifier(x, y, rand_state, use_tuning, nsplits, stdout=stdout)
        classifiers.append(svm)
        explainers.append(Explainer(svm, masker=Independent(x_t)))
    if use_xgb:
        names.append("XGBoost")
        xgb = xgb_classifier(x, y, rand_state, use_tuning, nsplits, stdout=stdout)
        classifiers.append(xgb)
        explainers.append(Explainer(xgb))
    if use_knn:
        names.append("K-Nearest Neighbors")
        knn = knn_classifier(x, y, rand_state, use_tuning, nsplits, stdout=stdout)
        classifiers.append(knn)
        explainers.append(Explainer(knn.predict, x_t))
    if use_mlp:
        names.append("Multi-Layer Perceptron")
        mlp = mlp_classifier(x, y, rand_state, use_tuning, nsplits, stdout=stdout)
        classifiers.append(mlp)
        explainers.append(Explainer(mlp.predict, x_t))

    if len(classifiers) == 0:
        stdout.write("No classifiers were executed. Exiting...")
        return

    for name, explainer in zip(names, explainers):
        stdout.write(f"Calculating SHAP-scores for {name} using {explainer.__class__.__name__}")
        shaps = explainer(x_t).values
        # Random Forest, unlike the other classifiers returns a matrix of shap values for each class (0 and 1). Other
        # classifers return only for the 1 class. Therefore, shape should be [samples x features], but for Random Forest,
        # it happens to be [samples x features x labels]. This code selects the appropriate label (for case = 1)
        if len(shaps.shape) == 3:
            shaps = shaps[:, :, 1]

        shap_df = pd.DataFrame(shaps, columns=x_t.columns, index=x_t.index).join(input_df.loc[x_t.index, id_column])
        shap_path = os.path.join(output_dir, f"{stem}_SHAP-Scores-{name.replace(' ', '-')}.csv")
        shap_df.to_csv(shap_path, index=False)
        stdout.write(f"Saved {name} SHAP scores to {shap_path}")
        model_shaps.append(shaps)

    names.append("Voting Classifier")
    voting = voting_classifier(x, y, voting_type, names, classifiers, stdout=stdout)
    classifiers.append(voting)

    metrics = None
    predictions: list[Series] = []
    # keep original id column and y column for testing data
    pred_df: DataFrame = input_df.loc[:, [id_column, y_label_col]].iloc[x_t.index, :]
    for name, classifier in zip(names, classifiers):
        stdout.write(f"Calculating Accuracy, ROC-AUC, and F1 scores for {name}")
        y_pred = classifier.predict(x_t)
        # first column = probablity of class being 0, second column = proba of being 1
        y_prob = classifier.predict_proba(x_t)[:, 1]
        df = DataFrame(
            [
                {
                    "Classifier": name,
                    "Accuracy": accuracy_score(y_t, y_pred),
                    "ROC-AUC": roc_auc_score(y_t, y_prob),
                    "F1": f1_score(y_t, y_pred, average="weighted"),
                }
            ]
        )
        roc_scores.append(roc_curve(y_t, y_prob))

        metrics = df if metrics is None else pd.concat([metrics, df], ignore_index=True)
        predictions.append(Series(y_pred, name=name, index=x_t.index))

        stdout.write(f"Calculating confusion matrix for {name}")
        confusion_df = pd.DataFrame(confusion_matrix(y_t, y_pred))  # size = labels x labels (2 x 2 for case/control)
        confusion_path = os.path.join(output_dir, f"{stem}_Confusion-Matrix-{name.replace(' ', '-')}.csv")
        confusion_df.to_csv(confusion_path, index=False)
        stdout.write(f"Saved {name} confusion matrix to {confusion_path}")
        model_confusion.append(confusion_df)

    metrics_path = os.path.join(output_dir, f"{stem}_Classifier-Metrics.csv")
    metrics.to_csv(metrics_path, index=False)
    stdout.write(f"Saved classifier metrics to {metrics_path}")

    pred_df = pd.concat([pred_df, *predictions])
    prediction_path = os.path.join(output_dir, f"{stem}_Classifier-Predictions.csv")
    pred_df.to_csv(prediction_path, index=False)
    stdout.write(f"Saved classifier predictions to {prediction_path}")

    if use_igenes:
        stdout.write("Calculating I-Genes score")

        for name, shap in zip(names, model_shaps):
            stdout.write(f"Calculating Herfindahl-Hirschman Indexes for {name}")
            # importances has dimensionality of [samples x features]. We want [1 x features]
            # so `importances` below is a row vector of dimension `features`
            shap = np.mean(shap, axis=0)  # 'flatten' the rows into their mean
            max = np.max(np.abs(shap))
            # skip normalization if all 0s
            normalized = shap / max if max != 0 else shap
            normalized_features_importances.append(normalized)
            # TODO: decide if axis=0 is necessary
            feature_hhi_weights.append(np.sum(np.square(normalized), axis=0))

        # np.sum() sums up all entries individually ([models x features] --> number) [[1, 2], [3, 4]] --> 10
        sum = np.sum(feature_hhi_weights)
        feature_hhi_weights = np.array(feature_hhi_weights) / (1 if sum == 0 else sum)
        initial_weights = 1 / len(classifiers)
        final_weights = initial_weights + (initial_weights * feature_hhi_weights)

        # Should become a list of dimension features
        igenes_scores = None
        for weight, importance in zip(final_weights, normalized_features_importances):
            result = weight * np.abs(importance)
            igenes_scores = igenes_scores + result if igenes_scores is not None else result

        # of size [features x models] --> list(zip(*)) converts a list of lists to a list of tuples that is the transpose
        transposed_importances = list(zip(*normalized_features_importances))
        # sums the expression direction of a featuer per model
        directions = [expression_direction(*scores) for scores in transposed_importances]

        igenes_df = DataFrame(
            {
                "Features": x_t.columns,
                "I-Genes Score": igenes_scores,
                "Expression Direction": directions,
            }
        )
        igenes_df["I-Genes Rankings"] = igenes_df["I-Genes Score"].rank(ascending=False).astype(int)

        igenes_path = os.path.join(output_dir, f"{stem}_I-Genes-Score.csv")
        igenes_df.to_csv(igenes_path, index=False)
        stdout.write(f"Saved igenes scores to {igenes_path}")

    if use_visualizations:
        stdout.write("Generating visualizations")
        num_selected_features = len(X.columns)

        def save_fig(fig: Figure, path):
            plt.tight_layout()
            fig.savefig(path)
            stdout.write(f"Saved to {path}")
            plt.close()

        def set_ax_labels(g: Axes, title="", x="", y=""):
            if title:
                g.set_title(title)
            if x:
                g.set_xlabel(x)
            if y:
                g.set_ylabel(y)

        def set_fig_labels(g: Figure, title="", x="", y=""):
            if title:
                g.suptitle(title)
            if x:
                g.supxlabel(x)
            if y:
                g.supylabel(y)

        for name, shap in zip(names, model_shaps):
            stdout.write(f"Swarm plot for {name}")
            # handles sizing automatically
            summary_plot(shap, x_t, plot_type="dot", show=False)
            ax = plt.gca()
            fig = plt.gcf()
            set_ax_labels(ax, title=f"{name} SHAP Scores", x="SHAP Value", y="Feature")
            save_fig(fig.figure, os.path.join(output_dir, f"{stem}_SHAP-Plot-{name.replace(' ', '-')}.png"))

        # for name, importances in zip(names, normalized_features_importances):
        #     stdout.write(f"Normalized feature importances plot for {name}")
        #     rf_imp = pd.DataFrame({"Features": X.columns, "Importances": importances})
        #     fig = sns.barplot(rf_imp, x="Importances", y="Features", fill=True, orient="h")
        #     set_ax_labels(fig.axes, title=f"{name} Normalized Feature Importances", x="Importance", y="Feature")
        #     save_fig(fig.figure, os.path.join(output_dir, f"{stem}_Normalized-Importance-Plot-{name.replace(' ', '-')}.png"))

        for name, confusion_df in zip(names, model_confusion):
            stdout.write(f"Confusion matrix plot for {name}")
            fig = sns.heatmap(confusion_df, cmap="Blues", annot=True)
            fig.figure.set_size_inches(4, 4)
            set_ax_labels(fig.axes, title=f"{name} Confusion Matrix", x="True Class", y="Predicted Class")
            save_fig(fig.figure, os.path.join(output_dir, f"{stem}_Confusion-Heatmap-{name.replace(' ', '-')}.png"))

        for name, (fpr, tpr, _) in zip(names, roc_scores):
            stdout.write(f"ROC Curve for {name}")
            plt.plot(fpr, tpr)
            ax = plt.gca()
            fig = plt.gcf()
            fig.set_size_inches(4, 4)
            set_ax_labels(ax, title=f"{name} ROC", x="False Positive Rate", y="True Positive Rate")
            save_fig(fig, os.path.join(output_dir, f"{stem}_ROC-Curve-{name.replace(' ', '-')}.png"))

        # for name, pred in zip(names, predictions):
        #     stdout.write(f"Diagnosis accuracy Sankey Chart for {name}")
        #     diagnosis_labels = [0, 1]
        #     ax_l = sankey(left=y_t, right=pred, leftLabels=diagnosis_labels, rightLabels=diagnosis_labels)
        #     ax_l.axis("on")
        #     ax_r = ax_l.twinx()
        #     for spine in ax_l.spines:
        #         ax_l.spines[spine].set_visible(False)
        #         ax_r.spines[spine].set_visible(False)
        #     ax_l.set_xticks([])
        #     ax_r.set_xticks([])
        #     ax_l.set_yticks([])
        #     ax_r.set_yticks([])
        #     set_ax_labels(ax_l, title=f"{name} Sankey Plot", y="True Class")
        #     set_ax_labels(ax_r, y="Predicted Class")
        #     save_fig(ax_l.figure, os.path.join(output_dir, f"{stem}_Sankey-Prediction-Plot-{name.replace(' ', '-')}.png"))

        # if use_rf and rf:
        #     stdout.write("Tree graph for Random Forest estimator")
        #     fig, _ = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
        #     plot_tree(rf.estimators_[0], feature_names=X.columns, filled=True)
        #     set_fig_labels(fig, title="Random Forest Decision Tree")
        #     save_fig(fig, os.path.join(output_dir, f"{stem}_RF-Estimator-Graph.png"))

        stdout.write("Box plot for feature distributions")
        # may print invalid values if the value is 0 (since 0 is undefined on a log scale). can ignore
        fig = sns.catplot(data=melted_df, hue=y_label_col, y="Feature", x="Value", kind="box", aspect=2, log_scale=True)
        fig.figure.set_size_inches(7, 2 + num_selected_features * 0.5)
        set_fig_labels(fig.figure, title="Feature Distribution")
        save_fig(fig.figure, os.path.join(output_dir, f"{stem}_Feature-Value-Distribution-Box.png"))

        stdout.write("Strip plot for feature distributions")
        fig = sns.catplot(data=melted_df, hue=y_label_col, y="Feature", x="Value", kind="strip", aspect=2, log_scale=True)
        fig.figure.set_size_inches(7, 2 + num_selected_features * 0.5)
        set_fig_labels(fig.figure, title="Feature Distribution")
        save_fig(fig.figure, os.path.join(output_dir, f"{stem}_Feature-Value-Distribution-Strip.png"))

        stdout.write("Pairwise intra/inter feature correlation plot")
        pairwise_df = X.join(Y)
        fig = sns.PairGrid(data=pairwise_df, hue=y_label_col)
        fig.figure.set_size_inches(num_selected_features * 2, 2 + num_selected_features * 2)
        fig.map_offdiag(sns.scatterplot)
        fig.map_diag(sns.kdeplot, fill=True)
        # needed to push title to top
        fig.figure.suptitle("Feature Intracorrelations and Intercorrelations", y=1)
        save_fig(fig.figure, os.path.join(output_dir, f"{stem}_Feature-Correlation-Plot.png"))

        stdout.write("Intra/inter feature correlations heatmap")
        fig = sns.heatmap(correlation_df, cmap="Blues", annot=True, fmt=".2f")
        fig.figure.set_size_inches(num_selected_features * 0.8, num_selected_features * 0.8)
        set_fig_labels(fig.figure, title="Feature Correlations")
        save_fig(fig.figure, os.path.join(output_dir, f"{stem}_Feature-Correlation-Heatmap.png"))

        plt.close()

    stdout.write("Finished Feature Classification")


def main(
    stdout: StdOut,
    cgit_file: str,
    selected_features_file: str,
    output_dir: str,
    rand_state: int,
    test_size: float,
    n_splits: int,
    voting_type: str,
    use_tuning: bool,
    use_normalization: bool,
    use_igenes: bool,
    use_visualizations: bool,
    use_rf: bool,
    use_svm: bool,
    use_xgb: bool,
    use_knn: bool,
    use_mlp: bool,
):
    stdout.write(f"Reading DataFrame from {cgit_file}")
    input_df = pd.read_csv(cgit_file)

    selected_cols = None
    if selected_features_file:
        stdout.write(f"Reading significant features from {selected_features_file}")
        with open(selected_features_file, "r") as f:
            selected_cols = input_df.columns[input_df.columns.isin(f.read().splitlines())].tolist()

    classify_features(
        input_df=input_df,
        selected_features=selected_cols,
        rand_state=rand_state,
        test_size=test_size,
        use_normalization=use_normalization,
        use_tuning=use_tuning,
        nsplits=n_splits,
        use_rf=use_rf,
        use_svm=use_svm,
        use_xgb=use_xgb,
        use_knn=use_knn,
        use_mlp=use_mlp,
        voting_type=voting_type,
        use_visualizations=use_visualizations,
        use_igenes=use_igenes,
        output_dir=output_dir,
        stem=f"{Path(cgit_file).stem}_{datetime.now().strftime('%m-%d-%Y-%I-%M-%S-%p')}",
        stdout=stdout,
    )
