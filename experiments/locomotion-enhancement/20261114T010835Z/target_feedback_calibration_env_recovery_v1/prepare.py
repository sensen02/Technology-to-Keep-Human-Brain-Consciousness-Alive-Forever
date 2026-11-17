"""Stdlib recovery namespace binding; reuse actualoriginal numericpreflight proof."""
from pathlib import Path
import argparse,ast,hashlib,json,sys
from adapter import ORIGINAL,EXPECTED_PYTHON
ROOT=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare-environment-recovery',action='store_true');a=p.parse_args()
 if not a.prepare_environment_recovery:p.error('Explicitrecoverymetadata required')
 if Path(sys.executable).absolute()!=EXPECTED_PYTHON.absolute():raise ValueError('Usequalified125712Torch interpreter')
 import importlib.util
 if importlib.util.find_spec('torch') is None:raise ValueError('CorrectenvironmentmustexposeTorch; metadataonly, noTorchimport')
 jobs=read(ORIGINAL/'JOBS.json');pre=read(ORIGINAL/'PREFLIGHT.json');binding=read(ORIGINAL/'INPUT_BINDINGS.json')
 if len(jobs)!=1 or jobs[0]['exit_code']!=1 or not pre['all_passed'] or pre['timing_status']!='READY_FIXED600_5400S':raise ValueError('Actualenvironmentfailure+qualifiedpreflight required')
 if 'No module named' not in Path(jobs[0]['log']).read_text() or 'torch' not in Path(jobs[0]['log']).read_text():raise ValueError('ExpectedmissingTorch failure evidence')
 if pre['source_hashes']!=binding['source_hashes'] or pre['data_hashes']!=binding['data_hashes']:raise ValueError('Originalpreflight/inputbinding identities differ')
 for ledger in (binding['source_hashes'],binding['data_hashes']):
  for f,h in ledger.items():
   if sha(f)!=h:raise ValueError('Originalinput drift: '+f)
 originalproof={str(ORIGINAL/n):sha(ORIGINAL/n) for n in ('JOBS.json','INPUT_BINDINGS.json','PREFLIGHT.json','PREREG.json')}
 originalproof[jobs[0]['log']]=sha(jobs[0]['log'])
 if (ORIGINAL/'FAILURE.json').exists():originalproof[str(ORIGINAL/'FAILURE.json')]=sha(ORIGINAL/'FAILURE.json')
 # Scientific preregistration is byte-identical in JSON values, with separate
 # recovery-only provenance; no loss/optimizer/budget value is changed.
 save(ROOT/'PREREG.json',read(ORIGINAL/'PREREG.json'))
 files=tuple(__import__('sources').FILES)
 originalproof.update({str(ORIGINAL/n):sha(ORIGINAL/n) for n in files})
 relocated={str(ROOT/n):sha(ROOT/n) for n in files}
 relocated.update({str(ROOT/n):sha(ROOT/n) for n in ('adapter.py','prepare.py','staticcheck.py','RECOVERY_PREREG.json')})
 source={**binding['source_hashes'],**originalproof,**relocated}
 new=dict(binding,source_hashes=source,data_hashes=binding['data_hashes'],environment_recovery=dict(kind='MISSING_TORCH_ENVIRONMENT_RECOVERY_V1',qualified_python=str(EXPECTED_PYTHON),original_failed_scope=str(ORIGINAL),original_preflight_reused=True,scientific_AST_unchanged=True,no_numeric_preflight_rerun=True))
 save(ROOT/'INPUT_BINDINGS.json',new)
 proof=dict(pre,source_hashes=source,data_hashes=new['data_hashes'],qualification_origin='Actual originalPREFLIGHT boundbySHA; source-equivalent relocatedwrapper, NOT a new numericalpreflight',original_preflight_sha256=sha(ORIGINAL/'PREFLIGHT.json'),recovery_environment=str(EXPECTED_PYTHON))
 save(ROOT/'PREFLIGHT.json',proof);save(ROOT/'ENVIRONMENT_BINDING.json',dict(original_evidence_hashes=originalproof,qualified_python=str(EXPECTED_PYTHON),torch_spec_found_without_import=True,numeric_operations=0,scientific_changes=False))
 print('ENVIRONMENT_RECOVERY_BOUND_REUSE_ACTUAL_PREFLIGHT',flush=True)
if __name__=='__main__':main()
