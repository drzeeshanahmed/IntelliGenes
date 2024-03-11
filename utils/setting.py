# UI Libraries
import os
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QPushButton,
    QFileDialog,
)

# System libraries
from typing import Any


class Setting:
    def __init__(self, name: str, default_value):
        self.name = name
        self.value = default_value
        self.default_value = default_value

    def set(self, value):
        self.value = value

    def reset(self):
        self.value = self.default_value

    def widget(self) -> QWidget:
        pass


class Group(Setting):
    def __init__(self, name: str, settings: list[Setting]):
        super().__init__(name, None)
        self.settings = settings

    def widget(self) -> QWidget:
        widget = QGroupBox(self.name)
        container_layout = QVBoxLayout()
        container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        container_layout.setSpacing(3)
        container_layout.setContentsMargins(10, 5, 10, 5)
        widget.setLayout(container_layout)

        for setting in self.settings:
            container_layout.addWidget(setting.widget())

        return widget


class Config:
    def __init__(self, settings: list[Setting]):
        self.settings = settings

    def get(self, name: str) -> Any | None:
        for setting in self.settings:
            if setting.name == name:
                return setting.name
            elif isinstance(setting, Group):
                for s in setting.settings:
                    if s.name == name:
                        return s.value
        return None

    def widget(self) -> QWidget:
        widget = QWidget()
        settings_layout = QHBoxLayout()
        settings_layout.setContentsMargins(0, 0, 0, 0)

        for setting in self.settings:
            settings_layout.addWidget(setting.widget())

        settings_layout.setSpacing(30)
        widget.setLayout(settings_layout)
        return widget

    def reset_settings(self):
        for setting in self.settings:
            if not isinstance(setting, Group):
                setting.reset()
            else:
                for s in setting.settings:
                    s.reset()


class IntSetting(Setting):
    def __init__(self, name: str, default_value: int, min: int, max: int, step: int):
        super().__init__(name, default_value)
        self.min = min
        self.max = max
        self.step = step

    def widget(self):
        widget = QWidget()
        container_layout = QHBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(container_layout)

        sb = QSpinBox()
        sb.setValue(self.value)
        sb.setMinimumWidth(100)
        sb.setMinimum(self.min)
        sb.setMaximum(self.max)
        sb.setSingleStep(self.step)
        sb.valueChanged.connect(self.set)

        container_layout.addWidget(QLabel(self.name))
        container_layout.addStretch(1)
        container_layout.addWidget(sb)

        return widget


class FloatSetting(Setting):
    def __init__(
        self, name: str, default_value: float, min: float, max: float, step: int
    ):
        super().__init__(name, default_value)
        self.min = min
        self.max = max
        self.step = step

    def widget(self):
        widget = QWidget()
        container_layout = QHBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(container_layout)

        sb = QDoubleSpinBox()
        sb.setValue(self.value)
        sb.setMinimumWidth(100)
        sb.setMinimum(self.min)
        sb.setMaximum(self.max)
        sb.setSingleStep(self.step)
        sb.valueChanged.connect(self.set)

        container_layout.addWidget(QLabel(self.name))
        container_layout.addStretch(1)
        container_layout.addWidget(sb)

        return widget


class BoolSetting(Setting):
    def __init__(self, name: str, default_value: bool):
        super().__init__(name, default_value)

    def widget(self):
        widget = QWidget()
        container_layout = QHBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(container_layout)

        cb = QCheckBox()
        cb.setChecked(self.value)
        cb.stateChanged.connect(lambda s: self.set(s == Qt.CheckState.Checked.value))

        container_layout.addWidget(QLabel(self.name))
        container_layout.addStretch(1)
        container_layout.addWidget(cb)

        return widget


class StrChoiceSetting(Setting):
    def __init__(self, name: str, default_value: int, options: list[str]):
        super().__init__(name, default_value)
        self.options = options

    def widget(self):
        widget = QWidget()
        container_layout = QHBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(container_layout)

        cb = QComboBox()
        cb.addItems(self.options)
        cb.setCurrentText(self.value)
        cb.currentTextChanged.connect(self.set)

        container_layout.addWidget(QLabel(self.name))
        container_layout.addStretch(1)
        container_layout.addWidget(cb)

        return widget


class TextFileSetting(Setting):
    def __init__(self, name: str, default_value: str):
        super().__init__(name, default_value)

    def widget(self):
        widget = QWidget()
        container_layout = QVBoxLayout()
        container_layout.setSpacing(0)
        container_layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(container_layout)

        button_layout = QHBoxLayout()
        path_layout = QHBoxLayout()
        path_layout.setAlignment(Qt.AlignmentFlag.AlignRight)

        pb = QPushButton("Reset" if self.value else "Select File")
        label = QLabel(
            os.path.basename(self.value) if self.value else "(No file selected)"
        )
        pb.clicked.connect(lambda: self.chooseFile(pb, label))

        button_layout.addWidget(QLabel(self.name))
        button_layout.addStretch(1)
        button_layout.addWidget(pb)
        path_layout.addWidget(label)

        container_layout.addLayout(button_layout)
        container_layout.addLayout(path_layout)

        return widget

    def chooseFile(self, pb: QPushButton, label: QLabel):
        if self.value:
            self.value = ""
        else:
            filename, ok = QFileDialog.getOpenFileName(
                parent=pb,
                caption="Select a file",
                dir="",
                filter="Text (*.txt)",
                selectedFilter="",
            )
            self.value = filename

        pb.setText("Reset" if self.value else "Select File")
        label.setText(self.value if self.value else "(No file selected)")
