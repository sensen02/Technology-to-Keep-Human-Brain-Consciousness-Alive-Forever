"""Strict JSON and crash-safe, same-filesystem artifact publication."""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


def strict_loads(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('nonfinite JSON: ' + value)
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('nonfinite JSON number: ' + value)
        return number
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite, parse_float=finite_float)


def encoded(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, ensure_ascii=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


class ArtifactStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('artifact path escapes store')
        return path

    def write_bytes(self, name, content):
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix='.pending-', dir=target.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, target)
            directory = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return target

    def write_json(self, name, value):
        return self.write_bytes(name, encoded(value))

    def read_json(self, name, default=None):
        path = self.path(name)
        return strict_loads(path.read_bytes()) if path.exists() else default
