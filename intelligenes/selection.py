# Data Manipulation libraries
import pandas as pd
from pandas import DataFrame, Series

# Machine Learning libraries
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE
from sklearn.tree import DecisionTreeClassifier

# Misc System libraries
from datetime import datetime
import os
from pathlib import Path

# Utilities libraries
from utils.stdout import StdOut


def recursive_elim(
    x: DataFrame, y: Series, rand_state: int, features_col: str, ranking_col: str
) -> DataFrame:
    e = DecisionTreeClassifier(random_state=rand_state)
    rfe = RFE(estimator=e, n_features_to_select=1).fit(x, y)

    df = pd.DataFrame(
        {
            "features": x.columns,
            "rank": rfe.ranking_,
        },
    )
    # Since we only select 1 feature above, only that feature will have rank '1'
    # Every other feature will have a sequential rank. So here, we select the top 10% of features
    # This is done so we have an explicit order of the features.
    return df.rename(columns={"features": features_col, "rank": ranking_col})


def pearson(x: DataFrame, y: Series, features_col: str, p_value_col: str) -> DataFrame:
    df = pd.DataFrame(
        {
            "features": x.columns,
            # Independently calculate p-value for each predictor vs y
            # the second element of the return is the p-value
            "p-value": [pearsonr(x[col], y)[1] for col in x.columns],
        }
    )

    return df.rename(columns={"features": features_col, "p-value": p_value_col})


def chi2_test(
    x: DataFrame, y: Series, features_col: str, p_value_col: str
) -> DataFrame:
    chi = SelectKBest(score_func=chi2, k="all").fit(x, y)
    df = pd.DataFrame(
        {
            "features": x.columns,
            "p-value": chi.pvalues_,
        }
    )

    return df.rename(columns={"features": features_col, "p-value": p_value_col})


def anova(x: DataFrame, y: Series, features_col: str, p_value_col: str) -> DataFrame:
    anova = SelectKBest(score_func=f_classif, k="all").fit(x, y)
    df = pd.DataFrame(
        {
            "features": x.columns,
            "p-value": anova.pvalues_,
        }
    )

    return df.rename(columns={"features": features_col, "p-value": p_value_col})


def min_max_scalar(x: DataFrame) -> DataFrame:
    return pd.DataFrame(MinMaxScaler().fit_transform(x), columns=x.columns)


### Calculates the most relevant features from `input` needed to calculate
## Needs a Type and ID column, and returns the same DataFrame with only selected
## features present
def select_features(
    input_df: DataFrame,
    stdout: StdOut,
    test_size: int,
    rand_state: int,
    use_normalization: bool,
    use_rfe: bool,
    use_anova: bool,
    use_chi2: bool,
    use_pearson: bool,
    output_dir: str,
    stem: str,
) -> DataFrame:
    id_column = "ID"
    y_label_col = "Type"

    for c in [id_column, y_label_col]:
        if c not in input_df.columns:
            stdout.write(f"Invalid format: Missing column '{c}' in CIGT file.")
            return

    parsed_input_df = input_df.drop(columns=[id_column])
    X = parsed_input_df.drop(columns=[y_label_col])
    Y = parsed_input_df[y_label_col]

    stdout.write("Selecting Important Features")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x, _, y, _ = train_test_split(X, Y, test_size=test_size, random_state=rand_state)

    features_col = "Features"
    rfe_col = "RFE Rankings"
    anova_col = "ANOVA (p-value)"
    chi2_col = "Chi-Square Test (p-value)"
    pearson_col = "Pearson's Correlation (p-value)"

    if use_normalization:
        stdout.write("Normalizing DataFrame")
        x = min_max_scalar(x)

    results: tuple[list[DataFrame], list[Series]] = []

    if use_rfe:
        stdout.write("Recursive Feature Elimination")
        result = recursive_elim(x, y, rand_state, features_col, rfe_col)
        results.append((result, result[rfe_col] <= int(len(x.columns) * 0.1)))
    if use_anova:
        stdout.write("Analysis of Variance")
        result = anova(x, y, features_col, anova_col)
        results.append((result, result[anova_col] < 0.05))
    if use_chi2:
        stdout.write("Chi-Squared Test")
        result = chi2_test(x, y, features_col, chi2_col)
        results.append((result, result[chi2_col] < 0.05))
    if use_pearson:
        stdout.write("Pearson Correlation")
        result = pearson(x, y, features_col, pearson_col)
        results.append((result, result[pearson_col] < 0.05))

    if len(results) == 0:
        stdout.write("No selectors were used. Exiting...")
        return

    all = None
    selected_mask = None
    for result, mask in results:
        all = result if all is None else all.merge(result, on=features_col)
        selected_mask = mask if selected_mask is None else selected_mask & mask

    selected: DataFrame = all.loc[selected_mask]

    all_path = os.path.join(output_dir, f"{stem}_All-Features.csv")
    selected_path = os.path.join(output_dir, f"{stem}_Selected-Features.csv")

    all.to_csv(all_path, index=False)
    selected.to_csv(selected_path, index=False)
    stdout.write(f"Saved all feature rankings to {all_path}")
    stdout.write(f"Saved selected feature rankings to {selected_path}")

    selected_cigt_path = os.path.join(output_dir, f"{stem}_Selected-CIGT-File.csv")
    selected_columns = [id_column, y_label_col]
    significant_features = selected[features_col].tolist()
    selected_columns.extend(significant_features)

    selected_df = input_df[selected_columns]
    selected_df.to_csv(selected_cigt_path, index=False)
    stdout.write(f"Saved the selected features CIGT file to {selected_path}")

    stdout.write("Finished Feature Selection")

    return significant_features


def main(
    stdout: StdOut,
    cgit_file: str,
    output_dir: str,
    rand_state: int,
    test_size: float,
    use_normalization: bool,
    use_rfe: bool,
    use_pearson: bool,
    use_anova: bool,
    use_chi2: bool,
):
    stdout.write(f"Reading DataFrame from {cgit_file}")
    input_df = pd.read_csv(cgit_file)

    select_features(
        input_df=input_df,
        stdout=stdout,
        rand_state=rand_state,
        test_size=test_size,
        use_normalization=use_normalization,
        use_rfe=use_rfe,
        use_pearson=use_pearson,
        use_anova=use_anova,
        use_chi2=use_chi2,
        output_dir=output_dir,
        stem=f"{Path(cgit_file).stem}_{datetime.now().strftime('%m-%d-%Y-%I-%M-%S-%p')}",
    )
