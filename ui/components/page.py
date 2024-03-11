# UI libraries
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import SignalInstance


class Page(QWidget):
    def __init__(
        self,
        inputFile: SignalInstance,
        outputDir: SignalInstance,
        onTabSelected: SignalInstance,
    ):
        super().__init__()
        self.inputFileSignal = inputFile
        self.outputDirSignal = outputDir
        self.onTabSelected = onTabSelected
