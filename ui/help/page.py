# UI Libraries
from PySide6.QtCore import Qt, SignalInstance
from PySide6.QtWidgets import QVBoxLayout, QLabel

# Custom UI libraries
from ui.components.page import Page


class HelpPage(Page):
    def __init__(
        self,
        inputFile: SignalInstance,
        outputDir: SignalInstance,
        onTabSelected: SignalInstance,
    ) -> None:
        super().__init__(inputFile, outputDir, onTabSelected)
        self._layout = QVBoxLayout()
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(self._layout)

        label = QLabel(
            """
To learn more about IntelliGenes Desktop, you can visit the GitHub page (https://github.com/drzeeshanahmed/intelligenes/tree/intelligenes-gui).


Authors:
Rishabh Narayanan
William DeGroat
Dinesh Mendhe
Habiba Abdelhalim

Research Supervisor:
Dr. Zeeshan Ahmed
"""
        )

        self._layout.addWidget(label)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
