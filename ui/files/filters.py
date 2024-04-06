import re
from PySide6.QtCore import SignalInstance, Qt
from PySide6.QtWidgets import QPushButton, QLabel, QWidget, QVBoxLayout, QCheckBox


class FiltersWidget(QWidget):
    def __init__(self, onUpdated: SignalInstance) -> None:
        super().__init__()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(QLabel("Filters:"))
        self.setLayout(layout)

        self.regex = {
            "Feature Distributions": [r".*Feature-Value-Distribution.*\.png", r".*Collapsed-Feature-Values\.csv"],
            "Feature Correlations": [
                r".*Feature-Correlation.*\.png",
                r".*Collapsed-Feature-Values\.csv",
                r".*Feature-Correlations\.csv",
            ],
            "Classifier Metrics": [r".*Classifier-Metrics\.csv", r".*Classifier-Predictions\.csv", r".*I-Genes-Score\.csv"],
            "Selector Metrics": [r".*All-Features\.csv", r".*Selected-Features\.csv", r".*Selected-CIGT-File\.csv"],
            "Confusion Matrices": [r".*Confusion-Heatmap.*\.png", r".*Confusion-Matrix.*\.csv"],
            "ROC Curves": [r".*ROC-Curve.*\.png"],
            # "Sankey Plots": [r".*Sankey-Prediction-Plot.*\.png", r".*Classifier-Predictions\.csv"],
            "SHAP Scores": [r".*SHAP-Plot.*\.png", r".*SHAP-Scores.*\.csv"],
        }
        self.widgets = {k: QCheckBox(k) for k in self.regex.keys()}

        for _, widget in self.widgets.items():
            widget.setChecked(True)
            widget.stateChanged.connect(onUpdated)
            layout.addWidget(widget)

        check = QPushButton("Check All")
        uncheck = QPushButton("Uncheck All")
        check.clicked.connect(self.selectAll)
        uncheck.clicked.connect(self.deselectAll)
        layout.addWidget(check)
        layout.addWidget(uncheck)

    def matches(self, item: str) -> bool:
        for key, widget in self.widgets.items():
            if widget.isChecked():
                for pattern in self.regex.get(key):
                    if re.fullmatch(pattern, item):
                        return True
        return False

    def selectAll(self):
        for _, widget in self.widgets.items():
            widget.setChecked(True)

    def deselectAll(self):
        for _, widget in self.widgets.items():
            widget.setChecked(False)
