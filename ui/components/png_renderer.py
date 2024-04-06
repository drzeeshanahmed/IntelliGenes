# UI libraries
from PySide6.QtWidgets import (
    QLabel,
    QScrollArea,
    QSizePolicy,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QCheckBox,
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt


class ImageRenderer(QWidget):
    def __init__(self, path: str):
        super().__init__()
        self.scaleFactor = 1

        layout = QVBoxLayout()
        self.setLayout(layout)
        scroll_layout = QHBoxLayout()
        layout.addLayout(scroll_layout)
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        self.scrollArea = QScrollArea()
        scroll_layout.addWidget(self.scrollArea)

        self.label = QLabel()
        self.label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.label.setScaledContents(True)
        self.scrollArea.setWidget(self.label)

        self.zoom_in = QPushButton("Zoom In")
        self.zoom_out = QPushButton("Zoom Out")
        self.reset_zoom = QPushButton("Reset Zoom")
        self.fit_window = QCheckBox("Fit to Window")
        button_layout.addWidget(self.zoom_in)
        button_layout.addWidget(self.zoom_out)
        button_layout.addWidget(self.reset_zoom)
        button_layout.addWidget(self.fit_window)

        self.zoom_in.clicked.connect(lambda: self.zoom(1.25))
        self.zoom_out.clicked.connect(lambda: self.zoom(0.8))
        self.reset_zoom.clicked.connect(lambda: self.resetZoom())
        self.fit_window.clicked.connect(lambda: self.handleZoomToFit())

        pixmap = QPixmap(path)
        self.label.setPixmap(pixmap)
        self.label.adjustSize()
        self.resetZoom()

    def zoom(self, factor):
        MAX_ZOOM_IN = 2
        MAX_ZOOM_OUT = 0.5
        self.scaleFactor = max(MAX_ZOOM_OUT, min(MAX_ZOOM_IN, self.scaleFactor * factor))
        self.resize()

    def resetZoom(self):
        self.scaleFactor = 1
        self.resize()

    def resize(self):
        self.label.resize(self.label.pixmap().size() * self.scaleFactor)

    def handleZoomToFit(self):
        isFitToWindow = self.fit_window.checkState() == Qt.CheckState.Checked
        self.scrollArea.verticalScrollBar().setDisabled(isFitToWindow)
        self.scrollArea.horizontalScrollBar().setDisabled(isFitToWindow)
        self.reset_zoom.setDisabled(isFitToWindow)
        self.zoom_in.setDisabled(isFitToWindow)
        self.zoom_out.setDisabled(isFitToWindow)

        if isFitToWindow:
            size = self.label.pixmap().size().scaled(self.scrollArea.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self.label.resize(size)
        else:
            self.resetZoom()
