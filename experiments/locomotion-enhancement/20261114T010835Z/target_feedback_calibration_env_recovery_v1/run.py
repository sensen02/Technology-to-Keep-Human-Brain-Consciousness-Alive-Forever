"""Correct-qualified interpreter only, new outputnamespace, exactoriginalstages."""
import sys
from pathlib import Path
from adapter import load,EXPECTED_PYTHON
if Path(sys.executable).absolute()!=EXPECTED_PYTHON.absolute():raise RuntimeError('Environmentrecovery requires actualqualified125712Torch Python')
load('run.py',globals())
