# UI libraries
import os
from PySide6.QtCore import SignalInstance, Qt
from PySide6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFileDialog,
    QPushButton,
)

# Custom libraries
from ui.components.page import Page
from ui.components.table_editor import TableEditor


class InputPage(Page):
    def __init__(
        self,
        inputFile: SignalInstance,
        outputDir: SignalInstance,
        onTabSelected: SignalInstance,
    ) -> None:
        super().__init__(inputFile, outputDir, onTabSelected)

        self.rendered_widget = None
        self.file = ""
        self.dir = ""

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(layout)

        self.restart_button = QPushButton("Reset IntelliGenes")
        self.restart_button.clicked.connect(lambda: (self.inputFileSignal.emit(""), self.outputDirSignal.emit("")))

        input_layout = QHBoxLayout()
        self.content_layout = QVBoxLayout()
        output_layout = QHBoxLayout()

        input_btn = QPushButton("Select Input File")
        input_label = QLabel()
        input_layout.addWidget(input_btn)
        input_layout.addWidget(input_label)
        self.inputFileSignal.connect(self.handleSelectedFile)
        self.inputFileSignal.connect(lambda text: input_label.setText("No file selected" if not text else text))
        output_btn = QPushButton("Select Output Location")
        output_label = QLabel()
        output_layout.addWidget(output_btn)
        output_layout.addWidget(output_label)
        self.outputDirSignal.connect(self.handleSelectedDir)
        self.outputDirSignal.connect(lambda text: output_label.setText("No directory selected" if not text else text))

        output_btn.clicked.connect(self.selectDirectory)
        input_btn.clicked.connect(self.selectFile)

        layout.addLayout(input_layout)
        layout.addLayout(self.content_layout)
        layout.addLayout(output_layout)
        layout.addWidget(self.restart_button)

    def selectFile(self):
        filename, ok = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select a CSV File",
            dir="",
            filter="CSV (*.csv)",
            selectedFilter="",
        )

        if filename:
            self.inputFileSignal.emit(filename)
            if not self.dir:
                self.outputDirSignal.emit(os.path.dirname(filename))

    def selectDirectory(self):
        dir = QFileDialog.getExistingDirectory()
        if dir:
            self.outputDirSignal.emit(dir)

    def handleSelectedDir(self, text: str):
        self.dir = text
        self.handleRestartButtonVisibility()

    def handleSelectedFile(self, path: str):
        self.file = path
        rendered_widget = None
        if not path:
            rendered_widget = QLabel("(Select an input file to preview)")
        elif path.endswith("csv"):
            rendered_widget = TableEditor(path, lambda filename: self.inputFileSignal.emit(filename))
        else:
            rendered_widget = QLabel("Unsupported file type")

        if self.rendered_widget is not None:
            self.content_layout.replaceWidget(self.rendered_widget, rendered_widget)
            self.rendered_widget.deleteLater()
        else:
            self.content_layout.addWidget(rendered_widget)
        self.rendered_widget = rendered_widget
        self.handleRestartButtonVisibility()

    def handleRestartButtonVisibility(self):
        if self.dir or self.file:
            self.restart_button.setVisible(True)
        else:
            self.restart_button.setVisible(False)
