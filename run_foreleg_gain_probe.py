"""S157/P1 one full-body paired foreleg gain probe.

Derived from the frozen S154 wrapper.  Only the shared front-leg gain is
changed: LF and RF actuators are .45, all four hind/middle legs remain .5.
Raw decoded commands are preserved; applied ctrl is decoded*u by the explicit
90-channel mapping.  This module is inert until main().
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, hashlib, json, sys, types
ROOT=Path(__file__).resolve().parent
S154=ROOT/'run_activation_rate_probe.py'
S142=ROOT/'run_free_physical_prewarm_probe.py'
MAP=Path('/home/sensen/Desktop/engine修复验收/s150-leg-command-unloading/20261024T213608033385Z/S149_ACTUATOR_MAP.json')
S154_REFERENCE=Path('/home/sensen/Desktop/engine修复验收/s154-activation-rate/20261024T215751775489Z')
SELECTION=Path('/home/sensen/Desktop/engine修复验收/s145-secant-contact-preload/20261024T203438782961Z')
OUTPUT_PARENT=Path('/home/sensen/Desktop/engine修复验收/s157-foreleg-gain')
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
def mapping_and_gains():
    rows=json.loads(MAP.read_text())
    assert len(rows)==90 and [r['actuator'] for r in rows]==list(range(90))
    counts={leg:sum(r['leg']==leg for r in rows) for leg in ('lf','rf','lm','rm','lh','rh')}
    assert counts=={'lf':15,'rf':15,'lm':15,'rm':15,'lh':15,'rh':15}
    # Explicit mapping assertion: every LF/RF index is known, not selected by name.
    front=[r['actuator'] for r in rows if r['leg'] in ('lf','rf')]
    assert len(front)==30 and len(set(front))==30
    gains=[.45 if r['leg'] in ('lf','rf') else .5 for r in rows]
    return rows,gains,counts

def prepare(selection):
    assert Path(selection).resolve()==SELECTION.resolve()
    rows,gains,counts=mapping_and_gains()
    s154=inert(S154,'_s157_s154')
    # S154 preparation binds the original S149 selection and frozen source chain.
    s142,binding,_,_=s154.prepare(selection)
    binding['hashes'].update({str(p):sha(p) for p in [MAP,Path(__file__).resolve(),S154_REFERENCE/'SOURCE_FREEZE/PRIVATE_RUNNER_SOURCE.py']})
    prior=s142.private_source; reasons=s142.stop_reasons
    module_source,_=replace_exact(S142.read_text(),[
      ("protocol='S142 ONE bounded free physical diagnostic; no assistance or activation assignment'",
       "protocol='S157 ONE bounded shared front foreleg gain contrast versus S154; no assistance or activation assignment', EXPERIMENT='FORELEG_GAIN', candidate=False, S119_same_condition_candidate=False, foreleg_gain_intervention=True")])
    s142=inert(S142,'_s157_private_s142',module_source); s142.stop_reasons=reasons
    gains_repr=repr(gains); map_repr=repr(rows)
    def transform(text):
        source,changes=prior(text)
        pre=("    # S157 derived per-actuator gain vector; mapping is explicit and archived.\n"
             f"    foreleg_mapping = {map_repr}\n    foreleg_gains = np.asarray({gains_repr}, dtype=float)\n"
             "    assert foreleg_gains.shape == (90,) and np.count_nonzero(foreleg_gains==.45)==30 and np.count_nonzero(foreleg_gains==.5)==60\n"
             "    atomic_json(out/'S149_ACTUATOR_MAP.json', dict(source=str(foreleg_mapping_path), source_sha256=foreleg_mapping_sha256, counts=foreleg_counts, rows=foreleg_mapping))\n"
             "    atomic_npz(out/'DERIVED_GAIN_CONFIG.npz', gains=foreleg_gains)\n")
        source,extra=replace_exact(source,[
          ("    atomic_npz(out/'DIRECTION.npz', weights=weights, **arms,",pre+"    atomic_npz(out/'DIRECTION.npz', weights=weights, **arms,"),
          ('            before = numeric_model_snapshot(m)','            before = numeric_model_snapshot(m)'),
          ('                d.ctrl[:] = 0.5 * u','                d.ctrl[:] = foreleg_gains * u'),
          ('                applied=0.5*np.asarray(commands),decoded=np.asarray(commands),current=np.asarray(currents),','                applied=foreleg_gains[None,:]*np.asarray(commands),decoded=np.asarray(commands),current=np.asarray(currents),'),
        ])
        prefix=("foreleg_mapping_path="+repr(str(MAP))+"\nforeleg_mapping_sha256="+repr(sha(MAP))+"\nforeleg_counts="+repr(counts)+"\n"
                "s154_reference="+repr(str(S154_REFERENCE))+"\n"+source)
        ast.parse(prefix); compile(prefix,str(s142.ORIGINAL),'exec')
        assert prefix.count('foreleg_gains * u')==1 and 'd.ctrl[:] = 0.5 * u' not in prefix
        return prefix,changes+extra
    s142.private_source=transform
    source,changes=transform(s142.ORIGINAL.read_text())
    return s142,binding,source,changes

def main():
    p=argparse.ArgumentParser(); p.add_argument('--selection',type=Path,default=SELECTION); p.add_argument('--output',default='auto'); a=p.parse_args()
    out=OUTPUT_PARENT/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') if a.output=='auto' else Path(a.output)
    if not out.is_absolute() or out.exists(): raise ValueError('fresh absolute output only; no resume')
    s142,binding,source,changes=prepare(a.selection)
    old=sys.argv
    try: sys.argv=[str(Path(__file__).resolve()),'--output',str(out)]; s142.main()
    finally: sys.argv=old
if __name__=='__main__': main()
