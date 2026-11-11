"""Exactly four fixed controlled diagnostics; every child collected."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
HERE=Path(__file__).resolve().parent

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    reg=json.loads((HERE/'PREREG.json').read_text())
    hashes={str(HERE/n):sha(HERE/n) for n in ('PREREG.json','run_one.py','pipeline.py','verify_saved.py')}
    (HERE/'SOURCE_IDENTITY.json').write_text(json.dumps(hashes,indent=2)+'\n')
    results=[]
    for profile in reg['profiles']:
        with (HERE/(profile+'.log')).open('x') as log:
            r=subprocess.run([sys.executable,'-B',str(HERE/'run_one.py'),'--profile',profile],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
        print(json.dumps(dict(profile=profile,exit_code=r.returncode)),flush=True)
        if r.returncode:raise RuntimeError('Physical evaluator failed; retain evidence and inspect log')
        result=json.loads((HERE/profile/'RESULT.json').read_text());results.append(result)
        print(json.dumps(dict(profile=profile,gates=result['autonomous']['gates'],slip=result['autonomous']['slip_ratio_max'],yaw=result['yaw'],cycles=result['autonomous_cycles']['complete_per_leg'],diagnostic_eligible=result['eligible'])),flush=True)
    with (HERE/'VERIFY.log').open('x') as log:
        r=subprocess.run([sys.executable,'-B',str(HERE/'verify_saved.py')],stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps(dict(phase='independent_saved_verify',exit_code=r.returncode)),flush=True)
    if r.returncode:raise RuntimeError('saved verification failed')
    if any(sha(p)!=h for p,h in hashes.items()):raise ValueError('new source drift')
    (HERE/'ALL4_RESULT.json').write_text(json.dumps(dict(profiles=results,diagnostic_only=True,food_qualification=False,source_hashes=hashes),indent=2)+'\n')
if __name__=='__main__':main()
