# UI libraries
from PySide6.QtWidgets import QTextEdit
from PySide6.QtGui import QTextCursor


class PipelineConsole(QTextEdit):
    def __init__(self) -> None:
        super().__init__()
        self.clear()
        self.setUndoRedoEnabled(False)
        self.setMinimumWidth(500)
        self.setReadOnly(True)

    def addLine(self, line: str):
        self.moveCursor(QTextCursor.MoveOperation.End)
        self.insertPlainText(line)
        self.ensureCursorVisible()
