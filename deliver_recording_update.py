"""Strict allowlist for recording update; validates every delivered byte."""
from pathlib import Path
import hashlib,json,shutil
SRC=Path(__file__).resolve().parent
DST=Path('/home/sensen/Desktop/cell_wound_prototype/embodied_body')
FILES=[
'engine/electrode_frontend.py','run_electrode_frontend.py',
'engine/embodied/access_map.py','outputs/embodied_body/FRONTEND_REPAIR.md',
'engine/electrode_recording.py','run_electrode_recording.py',
'run_recording_physics_regression.py','run_frontend_regression.py',
'engine/electrode_sorting.py','run_electrode_sorting.py',
'outputs/embodied_body/electrode_recording.json',
'outputs/embodied_body/electrode_recording.png',
'outputs/embodied_body/electrode_sorting.json',
'outputs/embodied_body/electrode_sorting.png',
'outputs/embodied_body/RECORDING_CORRECTIONS.md',
'outputs/embodied_body/RECORDING_MEMORY_INCIDENT.md',
'outputs/embodied_body/RECORDING_ARCHITECTURE.md',
'outputs/embodied_body/RECORDING_INDEPENDENT_AUDIT.md',
'outputs/embodied_body/VIRTUAL_BODY_RECORDING_DESIGN.md',
'deliver_recording_update.py']
def main():
    missing=[r for r in FILES if not (SRC/r).is_file()]
    if missing: raise RuntimeError('Required outputs missing: '+str(missing))
    manifest=[]
    for rel in FILES:
        src=SRC/rel; dst=DST/rel
        if src.is_symlink(): raise RuntimeError('Symlink refused: '+rel)
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        assert src.read_bytes()==dst.read_bytes()
        manifest.append({'path':str(dst),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'bytes':dst.stat().st_size})
    out=DST/'RECORDING_UPDATE_MANIFEST.json'
    out.write_text(json.dumps({'status':'hand-built research prototype; no biological validation','files':manifest},indent=2))
    print('Hash-verified',len(manifest),'files:',out)
if __name__=='__main__': main()
