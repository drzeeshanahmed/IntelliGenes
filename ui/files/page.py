# UI libraries
import os
from PySide6.QtCore import QDir, Signal, Qt, SignalInstance
from PySide6.QtWidgets import (
    QVBoxLayout,
    QListWidget,
    QListWidgetItem,
    QLabel,
    QFileDialog,
)

# Custom UI libraries
from ui.components.page import Page
from ui.components.png_renderer import ImageRenderer
from ui.components.table_editor import TableEditor


class OutputFilesPage(Page):
    selectedFile = Signal(str)

    def __init__(
        self,
        inputFile: SignalInstance,
        outputDir: SignalInstance,
        onTabSelected: SignalInstance,
    ) -> None:
        super().__init__(inputFile, outputDir, onTabSelected)
        self.path = None

        self.rendered_widget = None
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        label = QLabel("Output Location (Generated CSV and PNG files):")
        self._layout.addWidget(label)
        path_label = QLabel()
        self._layout.addWidget(path_label)

        self.list = QListWidget()
        self._layout.addWidget(self.list, 1) # stretch factor

        self.list.itemClicked.connect(
            lambda i: self.selectedFile.emit(i.data(Qt.ItemDataRole.UserRole))
        )

        self.outputDirSignal.connect(self.setOutputPath)
        self.outputDirSignal.connect(
            lambda text: path_label.setText(text if text else "Select an Output Location")
        )
        self.onTabSelected.connect(self.updateDirectoryWidgets)
        self.selectedFile.connect(self.handleSelectedFile)

    def setOutputPath(self, path: str):
        self.path = path

    def browseDirectory(self):
        dir = QFileDialog.getExistingDirectory()
        if dir:
            self.outputDirSignal.emit(dir)

    def updateDirectoryWidgets(self):
        self.list.clear()

        if not self.path:
            self.selectedFile.emit("")
            return

        dir = QDir(self.path)
        dir.setNameFilters(["*.csv", "*.png"])
        dir.setFilter(QDir.Filter.Files)
        files = dir.entryInfoList()

        for file in files:
            widget = QListWidgetItem(file.fileName())
            widget.setData(Qt.ItemDataRole.UserRole, file.absoluteFilePath())
            self.list.addItem(widget)

    def handleSelectedFile(self, path: str):
        if self.rendered_widget is not None:
            self.rendered_widget.deleteLater()

        if not path:
            self.rendered_widget = QLabel("Select a file to preview")
            self._layout.addWidget(self.rendered_widget)
        elif not os.path.exists(path):
            self.rendered_widget = QLabel("File not found")
            self._layout.addWidget(self.rendered_widget)
        else:
            if path.endswith("png"):
                self.rendered_widget = ImageRenderer(path)
            elif path.endswith("csv"):
                self.rendered_widget = TableEditor(path, lambda file: self.updateDirectoryWidgets())
            else:
                self.rendered_widget = QLabel("Unsupported file type")
            self._layout.addWidget(self.rendered_widget, 2)  # add with stretch factor

