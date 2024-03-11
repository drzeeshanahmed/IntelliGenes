# UI libraries
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QComboBox,
    QPushButton,
)

# Pipeline types
from intelligenes.intelligenes_pipelines import PipelineResult

class PipelineControls(QWidget):
    def __init__(
        self,
        pipelines: list[PipelineResult],
        run_button: QPushButton,
        combo_box: QComboBox,
    ) -> None:
        super().__init__()
        self.pipelines = pipelines
        self.run = run_button

        self._layout = QVBoxLayout()
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

        self.widget = None
        self.setIndex(combo_box.currentIndex())
        combo_box.currentIndexChanged.connect(self.setIndex)

        reset_button = QPushButton("Reset AI/ML Parameters")
        reset_button.clicked.connect(
            lambda: (
                pipelines[combo_box.currentIndex()][1].reset_settings(),
                self.setIndex(combo_box.currentIndex()),
            )
        )

        self._layout.addWidget(combo_box)
        self._layout.addWidget(self.widget)
        self._layout.addWidget(run_button)
        self._layout.addWidget(reset_button)

    def setIndex(self, index: int):
        _, config, _ = self.pipelines[index]
        old = self.widget
        self.widget = config.widget()
        if old is not None:
            self._layout.replaceWidget(old, self.widget)
            old.deleteLater()
