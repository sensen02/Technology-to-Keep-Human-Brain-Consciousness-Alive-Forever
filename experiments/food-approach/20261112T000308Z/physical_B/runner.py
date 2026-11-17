"""New behavior B-only wrapper, unchanged AB physical runner code object."""
import argparse
from runtime import entry
if __name__=='__main__':
 # Parent launches only after admission/preflight; inherited parser accepts B-only IDS.
 entry('runner.py','main')()
