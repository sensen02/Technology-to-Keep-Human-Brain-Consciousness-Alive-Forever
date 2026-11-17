"""Unchanged actualarchive metrics/independent audit; no behavior threshold changes."""
from common import module,ORIGINAL
_core=module('_fusion_recovery_exact_evaluator',ORIGINAL/'evaluate.py')
main=_core.main;audit=_core.audit
if __name__=='__main__':raise SystemExit('Use explicitflag run.py')
