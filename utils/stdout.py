# Miscellaneous system libraries
from collections import deque
import os


class StdOut:
    def __init__(self) -> None:
        self._istream = None
        self._ostream = None
        # once the stream is closed, collect extra lines in this list to be popped off
        self._reversed_final: deque = deque()
        self.open()

    def write(self, string: str):
        try:
            self._ostream.write(string + "\n")
        except Exception:
            pass

    def read(self):
        if not self._istream.closed:
            try:
                return self._istream.readline()
            except Exception:
                return None
        elif len(self._reversed_final) > 0:
            return self._reversed_final.popleft()
        else:
            return None

    def can_read(self):
        return not self._istream.closed or len(self._reversed_final) > 0

    def close(self):
        # Order is important. Closing ostream allows istream to recieve an EOF
        self._ostream.flush()
        self._ostream.close()
        for line in self._istream.readlines():
            self._reversed_final.append(line)

        self._istream.close()

    def open(self):
        self._reversed_final.clear()
        _r, _w = os.pipe()
        self._istream, self._ostream = os.fdopen(_r, "r"), os.fdopen(_w, "w", 1)
