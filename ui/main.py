# UI Libraries
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QTabWidget

# Miscellaneous system libraries
import sys

# Custom UI components
from ui.components.page import Page
from .input.page import InputPage
from .files.page import OutputFilesPage
from .pipeline.page import PipelinePage
from .help.page import HelpPage


class MainWindow(QMainWindow):
    # global state for input and output file
    # will be either a valid path or an empty string
    inputFile = Signal(str)
    outputDir = Signal(str)
    # Signals for when each page is selected
    inputPageSignal = Signal()
    pipelinePageSignal = Signal()
    filesPageSignal = Signal()
    demoPageSignal = Signal()


    def __init__(self):
        super().__init__()
        self.setWindowTitle("IntelliGenes 2")

        layout = QVBoxLayout()

        tabs: list[tuple[str, Page]] = [
            ("Data Manager", InputPage(self.inputFile, self.outputDir, self.inputPageSignal)),
            ("AI/ML Analysis", PipelinePage(self.inputFile, self.outputDir, self.pipelinePageSignal)),
            ("Visualization", OutputFilesPage(self.inputFile, self.outputDir, self.filesPageSignal)),
            ("Help", HelpPage(self.inputFile, self.outputDir, self.demoPageSignal)),
        ]
        def select_tab(index: int):
            tabs[index][1].onTabSelected.emit()
        
        self.inputFile.emit("")
        self.outputDir.emit("")

        tab_bar = QTabWidget()
        tab_bar.currentChanged.connect(select_tab)
        tab_bar.setTabPosition(QTabWidget.TabPosition.North)
        tab_bar.setDocumentMode(True)

        for name, widget in tabs:
            tab_bar.addTab(widget, name)
        tab_bar.setCurrentIndex(0)

        tab_bar.setLayout(layout)

        self.setCentralWidget(tab_bar)


def run():
    app = QApplication([])

    window = MainWindow()
    # resize to 80% of the available screen
    window.resize(app.primaryScreen().availableGeometry().size() * 0.8)
    window.show()

    sys.exit(app.exec())
