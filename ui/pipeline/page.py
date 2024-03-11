# UI libraries
from PySide6.QtCore import SignalInstance
from PySide6.QtWidgets import QVBoxLayout, QPushButton, QComboBox

# Custom UI libraries
from .controls import PipelineControls
from .console import PipelineConsole
from ui.components.page import Page

# Custom utilities
from utils.capture_output import CaptureOutput
from utils.stdout import StdOut

# Intelligenes pipelines
from intelligenes.intelligenes_pipelines import (
    select_and_classify_pipeline,
    classification_pipeline,
    feature_selection_pipeline,
    PipelineResult,
)


class PipelinePage(Page):
    def __init__(
        self,
        inputFile: SignalInstance,
        outputDir: SignalInstance,
        onTabSelected: SignalInstance,
    ) -> None:
        super().__init__(inputFile, outputDir, onTabSelected)

        self.stdout = StdOut()
        self.output = CaptureOutput(self.stdout)

        self.inputFilePath = None
        self.outputDirPath = None

        pipelines: list[PipelineResult] = [
            select_and_classify_pipeline(),
            feature_selection_pipeline(),
            classification_pipeline(),
        ]

        self.inputFileSignal.connect(
            lambda text: (self._setFile(text), self.reset(pipelines))
        )
        self.outputDirSignal.connect(
            lambda text: (self._setDir(text), self.reset(pipelines))
        )

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.combo_box = QComboBox()
        run_button = QPushButton("Execute Analysis")

        for name, _, _ in pipelines:
            self.combo_box.addItem(name)

        console = PipelineConsole()

        self.output.text.connect(console.setText)
        self.output.started.connect(lambda: run_button.setDisabled(True))
        self.output.finished.connect(lambda: run_button.setDisabled(False))

        run_button.clicked.connect(
            lambda: self.run(pipelines[self.combo_box.currentIndex()])
        )

        self.controls = PipelineControls(pipelines, run_button, self.combo_box)

        layout.addWidget(self.controls)
        layout.setStretch(0, 1)
        layout.addWidget(console)
        layout.setStretch(1, 2)

    def run_pipeline(self, pipeline: PipelineResult):
        # validate pipeline
        if not self.inputFilePath:
            self.stdout.write("Select an input file")
        elif not self.outputDirPath:
            self.stdout.write("Select an output location")
        else:
            pipeline[2](self.inputFilePath, self.outputDirPath, self.stdout)

    # Run process in a separate thread and capture output for the console
    def run(self, pipeline: PipelineResult):
        self.stdout.open()
        self.output.load_job(lambda: self.run_pipeline(pipeline))
        self.output.start()  # closes stdout when finished (need to reopen)

    def _setFile(self, text: str):
        self.inputFilePath = text

    def _setDir(self, text: str):
        self.outputDirPath = text

    def reset(self, pipelines: list[PipelineResult]):
        self.output.text.emit("")
        for _, config, _ in pipelines:
            config.reset_settings()
        self.combo_box.setCurrentIndex(0)
        self.controls.setIndex(0)
