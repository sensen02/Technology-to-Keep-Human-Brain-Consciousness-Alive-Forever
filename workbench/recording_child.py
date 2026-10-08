"""Resource limits before scientific imports; never writes legacy global outputs."""
import ctypes
import os
from pathlib import Path
import resource
import runpy
import signal
import sys


def main():
    root, output, parent = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    # Linux parent-death guard prevents orphan numerical workers after service crash.
    libc = ctypes.CDLL(None)
    if libc.prctl(1, signal.SIGKILL) != 0:
        raise RuntimeError('cannot install parent-death guard')
    if os.getppid() != parent:
        return
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))
    sys.path.insert(0, str(root))
    namespace = runpy.run_path(str(root / 'run_electrode_recording.py'), run_name='workbench_recording')
    # runpy functions retain their original globals, not necessarily returned mapping.
    namespace['main'].__globals__['OUT'] = output
    namespace['main']()


if __name__ == '__main__':
    main()
