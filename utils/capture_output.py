# UI Libraries
from PySide6.QtCore import Signal, QThread

# Custom utilities
from utils.stdout import StdOut

# System libraries
from threading import Thread
from typing import Callable


class Worker:
    def __init__(self, stdout: StdOut, callback: Callable[[], None]):
        super().__init__()
        self._process = None
        self._stdout = stdout
        self._callback = callback
        self._thread = None

    def start(self):
        # Memory Issue while using QThread for process, seems to work with normal Thread API
        if self._process:
            self._thread = Thread(target=self.run)
            self._thread.start()

    def load_job(self, process: Callable[[], None]):
        if self.is_alive():
            self._thread.join()
            self._thread = None

        self._process = process

    def run(self):
        try:
            self._process()
        except Exception as e:
            self._stdout.write(f"Execution of pipeline failed with message: {e}")
            self._stdout.write("Exiting...")
            pass
        try:
            if self._callback:
                self._callback()
        except Exception:
            pass

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()


class CaptureOutput(QThread):
    new_line = Signal(str)

    def __init__(self, stdout: StdOut):
        super().__init__()
        self._stdout = stdout
        # NOTE: very important to close the file descriptors to avoid memory leaks
        self._job = Worker(stdout, stdout.close)

    def load_job(self, job):
        self._job.load_job(job)

    def run(self):
        self._job.start()

        while self._job.is_alive() or self._stdout.can_read():
            line = self._stdout.read()
            if line is not None:
                self.new_line.emit(line)
