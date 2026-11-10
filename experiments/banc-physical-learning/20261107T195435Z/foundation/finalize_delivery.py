"""Byte-verified Desktop mirror; no neural/physics execution."""
import hashlib,json,shutil,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;DEST=Path('/home/sensen/Desktop/engine修复验收/banc-physical-learning/20261107T195435Z/foundation')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert json.loads((HERE/'FOUNDATION_RESULTS.json').read_text())['status']=='PASS_PRIMARY_FOUNDATION_CHECKS'
    assert (HERE/'INDEPENDENT_VERIFICATION.json').exists()
    assert json.loads((HERE/'INDEPENDENT_VERIFICATION.json').read_text())['status']=='PASS_INDEPENDENT_VERIFICATION'
    status=subprocess.run(['git','status','--short','--untracked-files=no'],cwd='/run/media/sensen/Data2/cell_wound_prototype',text=True,capture_output=True,check=True)
    (HERE/'TRACKED_STATUS.txt').write_text(status.stdout)
    files={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='DELIVERY_MANIFEST.json'}
    manifest={'workspace':str(HERE),'desktop':str(DEST),'files':files,'primary_neural_steps':2030,'independent_neural_steps':None,'physical_steps':0,'optimizer_updates':0,'mirror_scope':'foundation only'}
    (HERE/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    DEST.mkdir(parents=True,exist_ok=True)
    for p in HERE.iterdir():
        if p.is_file():shutil.copy2(p,DEST/p.name)
    assert all(sha(HERE/name)==sha(DEST/name) for name in list(files)+['DELIVERY_MANIFEST.json'])
    print(json.dumps({'files_verified':len(files)+1,'bytes':sum(x['bytes'] for x in files.values()),'desktop':str(DEST),'physical_steps':0,'optimizer_updates':0,'primary_neural_steps':2030},indent=2))
if __name__=='__main__':main()
