# UI libraries
from PySide6.QtWidgets import QTextEdit


class PipelineConsole(QTextEdit):
    def __init__(self) -> None:
        super().__init__()
        self.text = None
        self._update()
        self.setMinimumWidth(500)
        self.setReadOnly(True)

    def addLine(self, line: str):
        self.text = line if self.text is None else f"{self.text}\n{line}"
        self._update()

    def _update(self):
        self.setText("" if self.text is None else self.text)

    def clear(self):
        self.text = None
        self._update()
