"""Seal only new immutable analysis outputs; authenticate all read-only inputs."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    metrics=json.loads((HERE/'METRICS.json').read_text())
    changed=[p for p,h in metrics['input_hashes'].items() if sha(p)!=h]
    if changed:raise ValueError('input drift '+str(changed))
    files={str(HERE/n):dict(sha256=sha(HERE/n),bytes=(HERE/n).stat().st_size) for n in ('analyze.py','METRICS.json','README.md','RECOMMENDED_PLAN.json','seal.py')}
    report=dict(kind='READONLY_ANALYSIS_SEAL',files=files,input_files=len(metrics['input_hashes']),all_inputs_unchanged=True,Core_constructions=0,physics_integration_steps=0,optimizer_updates=0,new_training_datasets=0,scope='No new model, parameter-rate implementation, training preregistration or physical qualification; proposed plan only.')
    path=HERE/'ARTIFACT_SHA256.json'
    with path.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    assert all(sha(p)==v['sha256'] for p,v in files.items())
    print(json.dumps(dict(files=len(files),input_files=report['input_files'],changed=0,manifest_sha256=sha(path))))
if __name__=='__main__':main()
