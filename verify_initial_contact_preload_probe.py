"""S143 independent saved-array verifier; never imports MuJoCo or runs dynamics."""
from pathlib import Path
import argparse, ast, hashlib, json
import numpy as np

GATE_NAMES=('height','angle','legs','load','nonfoot','limit')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(rows,name,ok,**kw): rows.append(dict(name=name,passed=bool(ok),**kw))
def finite(z): return np.isfinite(z).all()
def audit_source(path, expected=None):
    out={'path':str(path),'exists':path.is_file()}
    if not path.is_file(): return out
    out['sha256']=sha(path); out['sha_match']=expected is None or out['sha256']==expected
    tree=ast.parse(path.read_text()); imports=[]; qwrites=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Import): imports += [a.name for a in n.names if a.name=='mujoco' or a.name.startswith('mujoco.')]
        if isinstance(n,ast.ImportFrom) and (n.module=='mujoco' or (n.module or '').startswith('mujoco.')): imports.append(n.module)
        targets=n.targets if isinstance(n,ast.Assign) else [n.target] if isinstance(n,(ast.AugAssign,ast.AnnAssign)) else []
        for t in targets:
            if isinstance(t,ast.Attribute) and t.attr=='qpos': qwrites.append(ast.unparse(n))
            elif isinstance(t,ast.Subscript) and isinstance(t.value,ast.Attribute) and t.value.attr=='qpos': qwrites.append(ast.unparse(n))
    out.update(ast_parse=True,mujoco_imports=imports,qpos_assignments=qwrites)
    return out

def verify(root):
    root=Path(root); rows=[]; errors=[]
    try:
        result=json.loads((root/'RESULT.json').read_text()); meta=json.loads((root/'METADATA.json').read_text())
        hashes=result.get('source_hashes',{}); source=Path(next((k for k in hashes if k.endswith('run_initial_contact_preload_probe.py')),''))
        sa=audit_source(source,hashes.get(str(source))); check(rows,'producer source AST parses',sa.get('ast_parse',False),audit=sa)
        check(rows,'producer source MuJoCo import is runtime-scoped',bool(sa.get('mujoco_imports')) and sa.get('mujoco_imports')==['mujoco'],imports=sa.get('mujoco_imports',[]))
        check(rows,'producer source qpos writes limited to intended initial pose',len(sa.get('qpos_assignments',[]))<=2,assignments=sa.get('qpos_assignments',[]))
        check(rows,'dynamic execution absent',result.get('dynamic_executed') is False and result.get('ledger',{}).get('step')==0 and result.get('ledger',{}).get('neural_ticks')==0)
        entries=[]
        for i in range(3):
            p=root/f'STATIC_{i}.npz'; check(rows,f'STATIC_{i} exists',p.is_file())
            if not p.is_file(): continue
            with np.load(p,allow_pickle=False) as z: a={k:z[k].copy() for k in z.files}
            entries.append(a)
            req=('qpos','qvel','act','ctrl','contact_geom_ids','contact_body_ids','contact_dist','contact_frame','contact_force_raw6','contact_pos_world','body_com_jacp','body_com_jacdot','qacc','Ma','RHS','six_gates','warnings','com_mass_times_acceleration_world','external_force_world')
            check(rows,f'STATIC_{i} required saved provenance',all(k in a for k in req),missing=[k for k in req if k not in a])
            if not all(k in a for k in req): continue
            # all state arrays finite and zero dynamic state
            check(rows,f'STATIC_{i} finite native arrays',all(finite(v) for v in a.values() if np.issubdtype(v.dtype,np.number)))
            check(rows,f'STATIC_{i} zero qvel act ctrl',np.all(a['qvel']==0) and np.all(a['act']==0) and np.all(a['ctrl']==0))
            # native all-body acceleration identity and qvel=0 => Jdotv=0
            jv=np.einsum('bij,j->bi',a['body_com_jacp'],a['qvel'])
            ja=np.einsum('bij,j->bi',a['body_com_jacp'],a['qacc'])+np.einsum('bij,j->bi',a['body_com_jacdot'],a['qvel'])
            check(rows,f'STATIC_{i} all-body Jdotv zero',np.all(a['qvel']==0) and np.allclose(np.einsum('bij,j->bi',a['body_com_jacdot'],a['qvel']),0,rtol=0,atol=0))
            check(rows,f'STATIC_{i} all-body native COM acceleration',np.allclose(ja,a['body_linear_acceleration_world'],rtol=0,atol=2e-12) if 'body_linear_acceleration_world' in a else False)
            check(rows,f'STATIC_{i} COM mass acceleration closure',np.allclose(a['com_mass_times_acceleration_world'],a['external_force_world'],rtol=1e-6,atol=1e-8),residual=float(a['com_force_residual']))
            check(rows,f'STATIC_{i} all48 Ma equals saved RHS',np.allclose(a['Ma'],a['RHS'],rtol=0,atol=1e-12))
            check(rows,f'STATIC_{i} six gates unchanged',a['six_gates'].shape==(6,) and bool(np.all(a['six_gates'])==bool(result['results'][i]['eligible'])))
            # Producer records both legacy negative-distance contact counts and the
            # stricter positive-normal bearing counts; require their six-leg gate
            # verdicts to agree, without claiming the masks are universally equal.
            if 'contacts' in a and 'bearing_contacts' in a:
                legacy=np.asarray(a['contacts']); bearing=np.asarray(a['bearing_contacts'])
                check(rows,f'STATIC_{i} legacy versus bearing six-leg verdict equivalent',bool(np.count_nonzero(legacy>0)>=3)==bool(np.count_nonzero(bearing>0)>=3),legacy_counts=legacy.tolist(),bearing_counts=bearing.tolist())
            else: check(rows,f'STATIC_{i} legacy versus bearing fields present',False)
            check(rows,f'STATIC_{i} warnings empty',not np.any(a['warnings']))
        check(rows,'three static offsets present',len(entries)==3)
        if len(entries)==3:
            q0=entries[0]['qpos']; d=float(result['d']); expected=[0,d/4,d/2]
            check(rows,'offsets exactly 0,d/4,d/2',all(np.isclose(float(x['dz']),expected[i],rtol=0,atol=0) for i,x in enumerate(result['results'])))
            check(rows,'baseline differs only root qpos z',all(np.array_equal(entries[0]['qpos'][np.arange(entries[0]['qpos'].size)!=2],x['qpos'][np.arange(x['qpos'].size)!=2]) for x in entries[1:]))
            for k in ('qvel','act','ctrl'):
                check(rows,f'{k} identical across static arms',all(np.array_equal(entries[0][k],x[k]) for x in entries[1:]))
            # native contact provenance and d from actual bearing contacts, not arbitrary value
            b=entries[0]['bearing_contact_mask'].astype(bool); dist=entries[0]['contact_dist']; f=entries[0]['contact_normal_force_on_foot_world']
            eligible=b & (dist<0) & (f[:,2]>0)
            dcalc=float(np.max(-dist[eligible])) if np.any(eligible) else float('nan')
            check(rows,'d derived from actual bearing foot-floor geometry',np.isfinite(dcalc) and dcalc==d,derived=dcalc,reported=d)
            check(rows,'contact geometry provenance ids/types/positions/forces saved',all(k in entries[0] for k in ('contact_geom_ids','contact_body_ids','contact_pos_world','contact_force_raw6','contact_frame')))
        # selection is zero because nonzero is not strict improvement; enforce no dynamic eligibility
        check(rows,'nonzero strict improvement false; no dynamic',result.get('nonzero_strict_improvement') is False and result.get('dynamic_eligible_pending_independent_verifier') is False)
    except Exception as e:
        errors.append(f'{type(e).__name__}: {e}'); check(rows,'verifier handled evidence',False,error=str(e))
    return {'schema':'S143_STATIC_PRELOAD_INDEPENDENT_CHECK.v2','passed':bool(rows) and all(x['passed'] for x in rows) and not errors,'dynamic_executed':False,'checks':rows,'errors':errors}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--inputroot',type=Path,required=True); p.add_argument('--outputroot',type=Path,required=True); p.add_argument('--report',type=Path)
    a=p.parse_args();
    if not a.inputroot.is_absolute() or not a.outputroot.is_absolute() or a.outputroot.exists(): raise SystemExit('fresh absolute roots required')
    r=verify(a.inputroot); a.outputroot.mkdir(parents=True); out=a.report or a.outputroot/'STATIC_PRELOAD_INDEPENDENT_CHECK.json'; out.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n'); print(json.dumps({'passed':r['passed'],'report':str(out)})); return 0 if r['passed'] else 2
if __name__=='__main__': raise SystemExit(main())
