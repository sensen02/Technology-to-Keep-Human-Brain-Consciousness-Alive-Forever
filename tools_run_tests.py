#!/usr/bin/env python3
"""Run top-level unittest test modules in isolated subprocesses.

Each module is a separate interpreter so process-wide settings (notably RLIMIT_AS)
from one test cannot affect another.  This tool records process output and timeout
state; it does not turn failures or timeouts into scientific conclusions.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable


def _module_paths(specs: Iterable[str], root: Path) -> list[Path]:
    values = list(specs)
    if not values:
        values = sorted(glob.glob(str(root / "test*.py")))
    paths: list[Path] = []
    for value in values:
        resolved = str(root / value) if not Path(value).is_absolute() else value
        matches = sorted(glob.glob(resolved)) if any(c in value for c in "*?[") else [resolved]
        if not matches:
            raise ValueError(f"module not found: {value}")
        for match in matches:
            p = Path(match)
            if not p.is_absolute():
                p = root / p
            if p.is_file() and p.suffix == ".py":
                paths.append(p.resolve())
            else:
                raise ValueError(f"module is not a Python file: {value}")
    # Preserve explicit order while removing duplicate paths.
    return list(dict.fromkeys(paths))


_FILE_RUNNER = """
import importlib.util
import sys
import unittest
from pathlib import Path
path = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(path.parent))
spec = importlib.util.spec_from_file_location('_isolated_test_module', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
suite = unittest.defaultTestLoader.loadTestsFromModule(module)
if suite.countTestCases() == 0:
    print('NO_TESTS: no unittest cases discovered; use the script-specific entry point', file=sys.stderr)
    raise SystemExit(5)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
"""


def run_one(path: Path, python: str, timeout: float, root: Path) -> dict:
    path = path.resolve()
    command = [python, "-c", _FILE_RUNNER, str(path)]
    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            command, cwd=str(root), capture_output=True, text=True,
            timeout=timeout, env=os.environ.copy(), check=False,
        )
        exit_code = completed.returncode
        stdout, stderr = completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = None
        stdout = _decode_output(exc.stdout)
        stderr = _decode_output(exc.stderr)
    except OSError as exc:
        exit_code = None
        stdout, stderr = "", f"LAUNCH_ERROR: {type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - started
    return {
        "module": path.name,
        "path": str(path.relative_to(root) if path.is_relative_to(root) else path),
        "command": command,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "timeout_seconds": timeout,
        "elapsed_seconds": round(elapsed, 6),
        "stdout": stdout,
        "stderr": stderr,
        "passed": (not timed_out and exit_code == 0),
        "no_tests": (exit_code == 5 and stderr.startswith("NO_TESTS:")),
        "launch_error": stderr.startswith("LAUNCH_ERROR:"),
    }


def _decode_output(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Python interpreter (default: current)")
    parser.add_argument("--timeout", type=float, default=60.0, help="per-module timeout in seconds")
    parser.add_argument("--modules", nargs="+", metavar="MODULE", help="module paths/globs (default: test*.py)")
    parser.add_argument("--output", metavar="JSON", help="write full JSON result to this file")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    root = Path.cwd().resolve()
    try:
        modules = _module_paths(args.modules or [], root)
        if not modules:
            raise ValueError("no test modules found")
    except ValueError as exc:
        parser.error(str(exc))
    results = [run_one(module, args.python, args.timeout, root) for module in modules]
    document = {
        "runner": "tools_run_tests.py",
        "python": args.python,
        "timeout_seconds": args.timeout,
        "modules": results,
        "summary": {
            "total": len(results),
            "passed": sum(item["passed"] for item in results),
            "failed": sum(not item["passed"] for item in results),
            "timed_out": sum(item["timed_out"] for item in results),
            "no_tests": sum(item["no_tests"] for item in results),
            "launch_errors": sum(item["launch_error"] for item in results),
        },
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    for item in results:
        status = ("PASS" if item["passed"] else "TIMEOUT" if item["timed_out"]
                  else "NO_TESTS" if item["no_tests"] else "LAUNCH_ERROR"
                  if item["launch_error"] else f"FAIL({item['exit_code']})")
        print(f"{status:8} {item['module']} [{item['elapsed_seconds']:.2f}s]")
    print(f"Summary: {document['summary']['passed']}/{document['summary']['total']} passed")
    return 0 if document["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
