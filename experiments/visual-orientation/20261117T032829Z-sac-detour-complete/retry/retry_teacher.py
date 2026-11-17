"""Repeat with the same teacher and a separate time budget."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import teacher

teacher.HERE = HERE
teacher.main()
