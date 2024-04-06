# Miscellaneous System libraries
from typing import Callable, TypeAlias

# Custom utilities
from utils.setting import (
    Config,
    Group,
    BoolSetting,
    IntSetting,
    FloatSetting,
    StrChoiceSetting,
    TextFileSetting,
)
from utils.stdout import StdOut

# IntelliGenes pipelines
from . import selection, classification, intelligenes

# (name, config, (input, output, stdout) -> None)
PipelineResult: TypeAlias = tuple[str, Config, Callable[[str, str, StdOut], None]]


def feature_selection_pipeline() -> PipelineResult:
    config = Config(
        [
            Group(
                "Parameters",
                [
                    IntSetting("Random State", 42, min=0, max=100, step=1),
                    FloatSetting("Test Size", 0.3, min=0, max=1, step=0.05),
                    BoolSetting("Normalize", False),
                ],
            ),
            Group(
                "Selectors",
                [
                    BoolSetting("Recursive Feature Elimination", True),
                    FloatSetting("RFE Top Percentile", 0.1, min=0, max=1, step=0.01),
                    BoolSetting("Pearson's Correlation", True),
                    FloatSetting("Pearson Alpha Threshold", 0.05, min=0, max=1, step=0.01),
                    BoolSetting("Analysis of Variance", True),
                    FloatSetting("ANOVA Alpha Threshold", 0.05, min=0, max=1, step=0.01),
                    BoolSetting("Chi-Squared Test", True),
                    FloatSetting("Chi-Squared Alpha Threshold", 0.05, min=0, max=1, step=0.01),
                ],
            ),
        ]
    )

    def run(
        cgit_file: str,
        output_dir: str,
        stdout: StdOut,
    ):
        selection.main(
            stdout=stdout,
            cgit_file=cgit_file,
            output_dir=output_dir,
            rand_state=config.get("Random State"),
            test_size=config.get("Test Size"),
            use_normalization=config.get("Normalize"),
            use_rfe=config.get("Recursive Feature Elimination"),
            use_pearson=config.get("Pearson's Correlation"),
            use_anova=config.get("Analysis of Variance"),
            use_chi2=config.get("Chi-Squared Test"),
            rfe_top_percentile=config.get("RFE Top Percentile"),
            anova_alpha=config.get("ANOVA Alpha Threshold"),
            chi2_alpha=config.get("Chi-Squared Alpha Threshold"),
            pearson_alpha=config.get("Pearson Alpha Threshold"),
        )

    return ("Feature Selection", config, run)


def classification_pipeline() -> PipelineResult:
    config = Config(
        [
            Group(
                "Parameters",
                [
                    IntSetting("Random State", 42, min=0, max=100, step=1),
                    FloatSetting("Test Size", 0.3, min=0, max=1, step=0.05),
                    IntSetting("N Splits", 5, min=1, max=20, step=1),
                    BoolSetting("Normalize", False),
                    BoolSetting("Tune", False),
                    BoolSetting("Calculate I-Genes", True),
                    BoolSetting("Create Visualizations", True),
                    TextFileSetting("Selected Features", ""),
                ],
            ),
            Group(
                "Classifiers",
                [
                    BoolSetting("Random Forest", True),
                    BoolSetting("Support Vector Machine", True),
                    BoolSetting("XGBoost", True),
                    BoolSetting("K-Nearest Neighbors", True),
                    BoolSetting("Multi-Layer Perceptron", True),
                    StrChoiceSetting("Voting", "soft", ["soft", "hard"]),
                ],
            ),
        ]
    )

    def run(
        cgit_file: str,
        output_dir: str,
        stdout: StdOut,
    ):
        classification.main(
            stdout=stdout,
            cgit_file=cgit_file,
            selected_features_file=config.get("Selected Features"),
            output_dir=output_dir,
            rand_state=config.get("Random State"),
            test_size=config.get("Test Size"),
            n_splits=config.get("N Splits"),
            use_normalization=config.get("Normalize"),
            use_tuning=config.get("Tune"),
            voting_type=config.get("Voting"),
            use_igenes=config.get("Calculate I-Genes"),
            use_visualizations=config.get("Create Visualizations"),
            use_rf=config.get("Random Forest"),
            use_svm=config.get("Support Vector Machine"),
            use_xgb=config.get("XGBoost"),
            use_knn=config.get("K-Nearest Neighbors"),
            use_mlp=config.get("Multi-Layer Perceptron"),
        )

    return ("Feature Classification", config, run)


def select_and_classify_pipeline() -> PipelineResult:
    config = Config(
        [
            Group(
                "Parameters",
                [
                    IntSetting("Random State", 42, min=0, max=100, step=1),
                    FloatSetting("Test Size", 0.3, min=0, max=1, step=0.05),
                    BoolSetting("Normalize for Selectors", False),
                    BoolSetting("Normalize for Classifiers", False),
                    IntSetting("N Splits", 5, min=1, max=20, step=1),
                    BoolSetting("Tune", False),
                    BoolSetting("Calculate I-Genes", True),
                    BoolSetting("Create Visualizations", True),
                ],
            ),
            Group(
                "Selectors",
                [
                    BoolSetting("Recursive Feature Elimination", True),
                    FloatSetting("RFE Top Percentile", 0.1, min=0, max=1, step=0.01),
                    BoolSetting("Pearson's Correlation", True),
                    FloatSetting("Pearson Alpha Threshold", 0.05, min=0, max=1, step=0.01),
                    BoolSetting("Analysis of Variance", True),
                    FloatSetting("ANOVA Alpha Threshold", 0.05, min=0, max=1, step=0.01),
                    BoolSetting("Chi-Squared Test", True),
                    FloatSetting("Chi-Squared Alpha Threshold", 0.05, min=0, max=1, step=0.01),
                ],
            ),
            Group(
                "Classifiers",
                [
                    BoolSetting("Random Forest", True),
                    BoolSetting("Support Vector Machine", True),
                    BoolSetting("XGBoost", True),
                    BoolSetting("K-Nearest Neighbors", True),
                    BoolSetting("Multi-Layer Perceptron", True),
                    StrChoiceSetting("Voting", "soft", ["soft", "hard"]),
                ],
            ),
        ]
    )

    def run(
        cgit_file: str,
        output_dir: str,
        stdout: StdOut,
    ):
        intelligenes.main(
            cgit_file=cgit_file,
            stdout=stdout,
            output_dir=output_dir,
            rand_state=config.get("Random State"),
            test_size=config.get("Test Size"),
            use_normalization_selectors=config.get("Normalize for Selectors"),
            use_normalization_classifiers=config.get("Normalize for Classifiers"),
            use_rfe=config.get("Recursive Feature Elimination"),
            use_pearson=config.get("Pearson's Correlation"),
            use_anova=config.get("Analysis of Variance"),
            use_chi2=config.get("Chi-Squared Test"),
            rfe_top_percentile=config.get("RFE Top Percentile"),
            anova_alpha=config.get("ANOVA Alpha Threshold"),
            chi2_alpha=config.get("Chi-Squared Alpha Threshold"),
            pearson_alpha=config.get("Pearson Alpha Threshold"),
            n_splits=config.get("N Splits"),
            use_tuning=config.get("Tune"),
            voting_type=config.get("Voting"),
            use_igenes=config.get("Calculate I-Genes"),
            use_visualizations=config.get("Create Visualizations"),
            use_rf=config.get("Random Forest"),
            use_svm=config.get("Support Vector Machine"),
            use_xgb=config.get("XGBoost"),
            use_knn=config.get("K-Nearest Neighbors"),
            use_mlp=config.get("Multi-Layer Perceptron"),
        )

    return ("Selection and Classification", config, run)
