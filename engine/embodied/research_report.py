"""Strict atomic research reports; do not silently stringify unsupported evidence."""
import json
import os
import tempfile
from pathlib import Path
import numpy as np


def atomic_report(path, report):
    path = Path(path)
    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    # Encode before touching the target, so a failed final report cannot destroy
    # the last complete checkpoint. Unsupported callbacks are errors, not strings.
    text = json.dumps(report, indent=2, allow_nan=False, default=encode) + '\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                dir=path.parent, prefix='.'+path.name+'.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def guard_snapshot(guard_report, status):
    """memory_guard.guarded_run yields a report FUNCTION, not a dict."""
    if not callable(guard_report):
        raise TypeError('expected guarded_run report callable')
    result = guard_report(result_status=status)
    if not isinstance(result, dict):
        raise TypeError('guard report must return dict')
    return result.copy()
