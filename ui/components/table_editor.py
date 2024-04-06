# Data manipulation libraries
from typing import Callable
import pandas as pd
import numpy as np

# UI libraries
from PySide6.QtWidgets import (
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QAbstractItemView,
    QLineEdit,
)


class TableEditor(QWidget):
    def __init__(self, path: str, onSave: Callable[[str], None]):
        super().__init__()
        self.path = path
        self.df = pd.read_csv(path)
        self.onSave = onSave
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.table = QTableWidget()
        layout.addWidget(self.table)

        # Prevent editing directly through cell (must use buttons below)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.updateTable()

        controls_layout = QHBoxLayout()
        layout.addLayout(controls_layout)

        # Update Cell Content
        edit_cell_layout = QVBoxLayout()
        self.cell_content = QLineEdit()
        self.save_cell_content = QPushButton("Update Cells")
        self.save_cell_content.clicked.connect(self.editCellContent)
        edit_cell_layout.addWidget(self.cell_content)
        edit_cell_layout.addWidget(self.save_cell_content)

        # Rename selected column
        edit_column_name = QVBoxLayout()
        self.column_name = QLineEdit()
        self.save_column_name = QPushButton("Rename Column")
        self.save_column_name.clicked.connect(self.renameColumn)
        edit_column_name.addWidget(self.column_name)
        edit_column_name.addWidget(self.save_column_name)

        # Save/Reset dataframe
        delete_col_row = QVBoxLayout()
        self.delete_col = QPushButton("Remove Columns")
        self.delete_col.clicked.connect(self.deleteSelectedColumns)
        self.delete_row = QPushButton("Remove Rows")
        self.delete_row.clicked.connect(self.deleteSelectedRows)
        delete_col_row.addWidget(self.delete_col)
        delete_col_row.addWidget(self.delete_row)

        # Save/Reset dataframe
        reset_save_layout = QVBoxLayout()
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.resetCSV)
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.handleSaveDataframe)
        reset_save_layout.addWidget(reset_button)
        reset_save_layout.addWidget(save_button)

        controls_layout.addLayout(edit_cell_layout)
        controls_layout.addLayout(edit_column_name)
        controls_layout.addLayout(delete_col_row)
        controls_layout.addLayout(reset_save_layout)

        self.table.itemSelectionChanged.connect(self.updateControlButtons)
        self.updateControlButtons()

    def updateControlButtons(self):
        ranges = self.table.selectedRanges()
        ROWS, COLS = self.df.shape
        # Edit multiple cells
        self.save_cell_content.setDisabled(len(ranges) == 0)

        # Rename column
        self.save_column_name.setDisabled(
            len(ranges) != 1
            or ranges[0].leftColumn() != ranges[0].rightColumn()
            or ranges[0].topRow() != 0
            or ranges[0].bottomRow() != ROWS - 1
        )

        # Only columns or only rows selected
        only_full_cols = True
        only_full_rows = True
        for r in ranges:
            if r.topRow() != 0 or r.bottomRow() != ROWS - 1:
                only_full_cols = False
            if r.leftColumn() != 0 or r.rightColumn() != COLS - 1:
                only_full_rows = False

        self.delete_col.setDisabled(len(ranges) == 0 or not only_full_cols)
        self.delete_row.setDisabled(len(ranges) == 0 or not only_full_rows)

    # Only full rows will be selected
    def deleteSelectedRows(self):
        ranges = self.table.selectedRanges()
        rows = set()
        for r in ranges:  # rows inclusive
            for i in range(r.topRow(), r.bottomRow() + 1):
                rows.add(i)

        self.df.drop(index=list(rows), inplace=True)
        self.updateTable()

    # Only full columns will be selected
    def deleteSelectedColumns(self):
        ranges = self.table.selectedRanges()
        cols = set()
        for r in ranges:  # rows inclusive
            for i in range(r.leftColumn(), r.rightColumn() + 1):
                cols.add(i)

        self.df.drop(columns=self.df.columns[list(cols)], inplace=True)
        self.updateTable()

    # Invariant: A single column will be selected if this method is invoked
    def renameColumn(self):
        text = self.column_name.text()
        column = self.table.selectedRanges()[0].leftColumn()
        if text and column not in self.df.columns:
            self.df.columns.values[column] = text
            self.column_name.clear()
            self.updateTable()

    def editCellContent(self):
        text = self.cell_content.text()
        if not text:
            return

        for item in self.table.selectedItems():
            t = self.df.dtypes.iloc[item.column()]
            try:
                if t == np.int_:
                    self.df.iloc[item.row(), item.column()] = int(text)
                elif t == np.float_:
                    self.df.iloc[item.row(), item.column()] = float(text)
            except Exception:
                continue

        self.cell_content.clear()
        self.updateTable()

    def resetCSV(self):
        self.df = pd.read_csv(self.path)
        self.updateTable()

    def handleSaveDataframe(self):
        filename, ok = QFileDialog.getSaveFileName(
            parent=self,
            caption="Save CSV File",
            dir=self.path,
            filter="CSV (*.csv)",
            selectedFilter="",
        )
        if filename:
            self.df.to_csv(filename, index=False)
            self.onSave(filename)

    def updateTable(self):
        self.table.clear()
        ROWS, COLS = self.df.shape
        self.table.setRowCount(ROWS)  # header
        self.table.setColumnCount(COLS)
        # To prevent eliding of data
        self.table.setWordWrap(False)
        self.table.setHorizontalHeaderLabels(self.df.columns)

        for r in range(ROWS):
            for c in range(COLS):
                widget = QTableWidgetItem(str(self.df.iloc[r, c]))
                self.table.setItem(r, c, widget)

        self.table.resizeColumnsToContents()
