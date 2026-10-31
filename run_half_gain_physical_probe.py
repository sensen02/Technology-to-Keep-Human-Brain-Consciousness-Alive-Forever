"""S149/P1 ONE <=10ms half-interface-gain physical probe.

Filesystem/AST preparation is inert. Parent invocation performs exactly one
actual run using the proven S146 initial pose and private S142 recorder. Raw
decoded commands are preserved; only applied ctrl is 0.5 * decoded u.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, hashlib, json, sys, types
ROOT=Path(__file__).resolve().parent
S146=ROOT/'run_secant_pose_physical_probe.py'
S142=ROOT/'run_free_physical_prewarm_probe.py'
OUTPUT_PARENT=Path('/home/sensen/Desktop/engine修复验收/s149-half-gain-physical')
S145_SELECTION=Path('/home/sensen/Desktop/engine修复验收/s145-secant-contact-preload/20261024T203438782961Z')
S146_REFERENCE=Path('/home/sensen/Desktop/engine修复验收/s146-secant-pose-free-physical/20261024T203645678947Z')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inert(path,name,source=None):
    m=types.ModuleType(name); m.__file__=str(path)
    exec(compile(path.read_text() if source is None else source,str(path),'exec'),m.__dict__); return m
def replace_exact(source,repls):
    changes=[]
    for old,new in repls:
        assert source.count(old)==1,(old,source.count(old))
        source=source.replace(old,new,1); changes.append({'old':old,'new':new})
    ast.parse(source); return source,changes

def preflight(selection):
    assert Path(selection).resolve()==S145_SELECTION.resolve()
    s146=inert(S146,'_s149_s146')
    binding=s146.preflight(selection)
    refs=[S146_REFERENCE/n for n in ('PREREG.json','INPUT_HASHES.json','COMPLETE.json','POSE_INITIALIZATION.npz','SECANT_SELECTION_BINDING.json','PARENT_S146_INDEPENDENT_CHECK.json','init80ms/TRAJECTORY.npz','init80ms/INITIALIZATION.npz','init80ms/NEURAL_COLD.npz','init80ms/NEURAL_COLD_META.json','init80ms/NEURAL_RELEASE_META.json','SOURCE_FREEZE/PRIVATE_RUNNER_SOURCE.py')]
    assert all(p.is_file() for p in refs)
    proof=json.loads((S146_REFERENCE/'SECANT_SELECTION_BINDING.json').read_text())
    assert float(proof['offset'])==binding['offset']
    binding['hashes'].update({str(p):sha(p) for p in [S146,Path(__file__).resolve(),*refs]})
    return binding

def prepare(selection):
    binding=preflight(selection)
    s146=inert(S146,'_s149_s146_transform')
    patched,_=replace_exact(S142.read_text(),[
      ("protocol='S142 ONE bounded free physical diagnostic; no assistance or activation assignment'", "protocol='S149 ONE <=10ms HALF_GAIN_PHYSICAL; no assistance or activation assignment', EXPERIMENT='HALF_GAIN_PHYSICAL', candidate=False, S119_same_condition_candidate=False, half_interface_gain=True"),
      ("physical_duration_ms=10,physical_steps_per_arm={'init80ms':400}", "physical_duration_ms=10,physical_steps_per_arm={'init80ms':400}"),
    ])
    s142=inert(S142,'_s149_private_s142',patched)
    base=s142.private_source
    def transform(text):
        source,changes=s146.private_source(text,base)
        repl=[
          ('                d.ctrl[:] = u','                d.ctrl[:] = 0.5 * u'),
          ('                applied=np.asarray(commands),decoded=np.asarray(commands),current=np.asarray(currents),','                applied=0.5*np.asarray(commands),decoded=np.asarray(commands),current=np.asarray(currents),'),
          ("physical_initial='same saved qpos; zero qvel/act/ctrl/external loads; no later pose writes'", "physical_initial='S146 identical proven NEWpose initialization; zero qvel/act/ctrl/external loads; no later pose writes', S119_same_condition_candidate=False, half_interface_gain=True, raw_decoded_preserved=True, applied_interface='all90 ctrl=0.5*decoded raw u after decode; no act overwrite; no tau change', sensory_history='S146 NEWpose real observation and PRE160 genuine neural rerun; no replay'"),
          ("str(ROOT/'run_secant_pose_physical_probe.py')]:", "str(ROOT/'run_secant_pose_physical_probe.py'), str(ROOT/'run_half_gain_physical_probe.py')]:"),
        ]
        source,extra=replace_exact(source,repl)
        prefix='s146_offset='+repr(binding['offset'])+'\n'+'s146_selection='+repr(binding)+'\n'+'s146_bindings='+repr(list(binding['hashes']))+'\n'+'s149_reference='+repr(str(S146_REFERENCE))+'\n'
        source=prefix+source; ast.parse(source)
        return source,changes+extra
    s142.private_source=transform
    original_reasons=s142.stop_reasons
    def reasons(row,qnew,m):
        import numpy as np
        from run_native_rk4_tick_consumption import INITIAL
        if not hasattr(reasons,'original'):
            with np.load(INITIAL,allow_pickle=False) as z: reasons.original=z['new_qpos'].copy()
        return original_reasons(row,reasons.original,m)
    s142.stop_reasons=reasons
    source,changes=transform(s142.ORIGINAL.read_text())
    assert 'd.ctrl[:] = 0.5 * u' in source and 'd.ctrl[:] = u' not in source
    assert 'applied=0.5*np.asarray(commands)' in source
    return s142,binding,source,changes

def main():
    p=argparse.ArgumentParser(); p.add_argument('--selection',type=Path,required=True); p.add_argument('--output',default='auto'); a=p.parse_args()
    out=OUTPUT_PARENT/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') if a.output=='auto' else Path(a.output)
    if not out.is_absolute() or out.exists(): raise ValueError('fresh absolute output only')
    s142,binding,source,changes=prepare(a.selection)
    assert binding['hashes']=={n:sha(n) for n in binding['hashes']}
    old=sys.argv
    try: sys.argv=[str(Path(__file__).resolve()),'--output',str(out)]; s142.main()
    finally: sys.argv=old
if __name__=='__main__': main()
