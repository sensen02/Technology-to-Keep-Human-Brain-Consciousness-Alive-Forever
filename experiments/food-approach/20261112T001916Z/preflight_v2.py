"""BeforeTRAIN recovery: generate PAIR_PREFLIGHT before frozen source hashing.
Original adapter/source/failure and completed dataset are preserved unchanged.
"""
from pathlib import Path
import ast
from adapt_source import build,once
HERE=Path(__file__).resolve().parent
def build_source():
 src=build('preflight.py')
 block=" from pair_checks import extra_checks\n checks.update(extra_checks(m))\n"
 if src.count(block)!=1:raise ValueError('exactlatepaircheckblock required')
 src=src.replace(block,'',1)
 marker=" paths=source_paths()+[HERE/'DATASET.json',HERE/'train.npz',HERE/'validation.npz',MOTOR]"
 src=once(src,marker,block+marker)
 return src
def main():
 ns={'__name__':'_paired_preflight_recovery_order_v2','__file__':str(HERE/'preflight_v2.py')};exec(compile(build_source(),str(HERE/'preflight_v2.py')+'::correct_paircheck_beforehash','exec'),ns)
 inherited_paths=ns['source_paths']
 def source_paths():return inherited_paths()+[HERE/'preflight_v2.py',HERE/'run_remaining_v2.py']
 ns['source_paths']=source_paths;ns['main']()
if __name__=='__main__':main()
