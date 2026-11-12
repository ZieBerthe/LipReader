import contextlib
import os
import sys


@contextlib.contextmanager
def suppress_stderr():
    """Silence stderr at the file-descriptor level (also catches C++ logs from MediaPipe)."""
    stderr_fd = sys.stderr.fileno()
    with open(os.devnull, "w") as devnull:
        old_stderr = os.dup(stderr_fd)
        try:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
        finally:
            os.dup2(old_stderr, stderr_fd)
            os.close(old_stderr)
