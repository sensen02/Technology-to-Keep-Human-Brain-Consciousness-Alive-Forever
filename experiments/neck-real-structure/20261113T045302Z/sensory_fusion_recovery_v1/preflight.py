"""Same original full27/micro9 numerical gates; corrected dataset checks run before these."""
from common import module,ORIGINAL
_core=module('_fusion_recovery_exact_preflight',ORIGINAL/'preflight.py')
main=_core.main
if __name__=='__main__':raise SystemExit('Use explicitflag run.py')
