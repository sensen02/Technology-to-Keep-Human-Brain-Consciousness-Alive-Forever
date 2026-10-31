"""S187: targeted attack on DOF 22 (LMCoxa_roll) - the only near-feasible row (m=1.32).

Objective = that single row's interval violation. Hard constraints: geometry gates. Root is
held at the S177 root-balanced point for the search; the best points are then re-checked with
a dim-3 root Newton and the full 42-row LP. No mj_step, no neural, no training.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, csv, json, shutil, hashlib
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import linprog
from root_newton_core import RootNewton
B_=Path('/home/sensen/Desktop/engine修复验收'); R=Path('/run/media/sensen/Data2/cell_wound_prototype')
XML=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
S154=B_/'s154-activation-rate/20261024T215751775489Z'; S166=B_/'s166-mapped-support/20261024T231418078692647Z'
S177=B_/'s177-root-newton-search/20261025T020500000000Z'
CAP, RESERVED = 900, 200
DEPTH_CAP=0.03442910871489635
TARGET_DOF=22
STATE=('qpos','qvel','act','ctrl','qfrc_applied','xfrc_applied'); OPT=('mocap_pos','mocap_quat','eq_active','plugin_state','userdata')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    def s(x):
        if isinstance(x,dict): return {str(k):s(y) for k,y in x.items()}
        if isinstance(x,(list,tuple)): return [s(y) for y in x]
        if isinstance(x,np.generic): x=x.item()
        if isinstance(x,float) and not np.isfinite(x): return None
        return x if isinstance(x,(int,float,bool,str)) or x is None else str(x)
    p.write_text(json.dumps(s(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
def extract(p,names):
    t=ast.parse(p.read_text()); return '\n'.join(ast.unparse(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==x)) for x in names)+'\n'
def qmul(a,b):
    aw,ax,ay,az=a; bw,bx,by,bz=b
    return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])
def qxyz(rx,ry,rz):
    cx,sx=np.cos(rx/2),np.sin(rx/2); cy,sy=np.cos(ry/2),np.sin(ry/2); cz,sz=np.cos(rz/2),np.sin(rz/2)
    return np.array([cx*cy*cz+sx*sy*sz, sx*cy*cz-cx*sy*sz, cx*sy*cz+sx*cy*sz, cx*cy*sz-sx*sy*cz])
def main(out):
    assert not out.exists(); out.mkdir(parents=True); (out/'SOURCE_FREEZE').mkdir()
    files=[Path(__file__).resolve(),XML,S154/'INPUT_HASHES.json',S154/'init80ms/TRAJECTORY.npz',S154/'init80ms/MODEL_PROVENANCE.json',
           S154/'init80ms/COM_CONTACT_METADATA.json',S166/'MAPPING.npz',S177/'ROOT_BALANCE_STATE.npz',S177/'ROOT_RESULT.json']
    old=json.loads((S154/'INPUT_HASHES.json').read_text()); old=old.get('hashes',old)
    h={}
    for p,v in old.items():
        if isinstance(v,str) and len(v)==64: assert sha(p)==v,'S154 binding'; h[p]=v
    h.update({str(p):sha(p) for p in files}); dump(out/'INPUT_HASHES.json',h)
    for i,(p,v) in enumerate(sorted(h.items())):
        t=out/'SOURCE_FREEZE'/f'{i:04d}_{Path(p).name}'; shutil.copyfile(p,t); assert sha(t)==v
    z=load(S154/'init80ms/TRAJECTORY.npz'); q0=z['qpos'][0].copy()
    proof=json.loads((S154/'init80ms/MODEL_PROVENANCE.json').read_text())
    meta=json.loads((S154/'init80ms/COM_CONTACT_METADATA.json').read_text())
    T=load(S166/'MAPPING.npz')['T']; st=load(S177/'ROOT_BALANCE_STATE.npz')
    p177=np.array(json.loads((S177/'ROOT_RESULT.json').read_text())['p'],float)
    tree=ET.parse(XML); _j=list(tree.getroot().find('worldbody').iter('joint'))
    assert _j[0].get('type')=='free'
    jn=[j.get('name') for j in _j[1:]]
    nat=np.array([[float(v) for v in j.get('range').split()] for j in _j[1:]])
    assert nat.shape==(42,2), nat.shape
    specs=[(R/'run_matched_state_counterfactual.py',('snapshot',)),(R/'run_local_pool_response.py',('moment_record',)),(R/'run_constraint_row_recording.py',('constraint_record',))]
    ns=dict(np=np,ah=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    exec(compile('\n'.join(extract(p,n) for p,n in specs),'<inert>','exec'),ns)
    calls=dict(model_load=0,forward=0,candidate_evaluations=0,inner_lp=0,root_newton_iterations=0,root_jacobians=0,cache_hits=0)
    import mujoco as mj
    calls['model_load']+=1; m=mj.MjModel.from_xml_path(str(XML))
    assert ns['snapshot'](m)==proof['before']
    m.actuator_dynprm[:,0]=.001; m.actuator_dynprm[:,1]=.0004
    m.opt.integrator=mj.mjtIntegrator.mjINT_RK4; m.opt.timestep=.000025
    assert ns['snapshot'](m)==proof['after']
    floors=set(meta['floor_geom_ids']); gleg={int(k):v for k,v in meta['geom_map'].items()}
    cache={}
    def fwd(q,a=None):
        d=mj.MjData(m); d.time=float(z['time'][0])
        for n in STATE: getattr(d,n)[:]=z[n][0]
        for n in OPT:
            if n in z: getattr(d,n)[:]=z[n][0]
            else: assert np.asarray(getattr(d,n)).size==0
        d.qpos[:]=q; d.qvel[:]=0
        act=np.zeros(90) if a is None else a
        d.act[:]=act; d.ctrl[:]=act; d.qacc_warmstart[:]=0
        mj.mj_forward(m,d); calls['forward']+=1
        c=dict(af=np.asarray(d.actuator_force).copy(),qacc=np.asarray(d.qacc).copy(),ad=np.asarray(d.act_dot).copy(),
               pas=np.asarray(d.qfrc_passive).copy(),bias=np.asarray(d.qfrc_bias).copy(),
               efc_D=np.asarray(d.efc_D).copy(),efc_aref=np.asarray(d.efc_aref).copy(),efc_type=np.asarray(d.efc_type).copy(),
               cdist=np.array([v.dist for v in d.contact]),cfric=np.array([v.friction.copy() for v in d.contact]).reshape(-1,5))
        c.update({k:np.asarray(v).copy() for k,v in ns['constraint_record'](mj,m,d,0).items()})
        c['mom']=np.asarray(ns['moment_record'](m,d)['actuator_moment']).copy()
        return c
    def build(qleg,p3):
        q=q0.copy(); q[7:]=qleg; q[0:2]=q0[0:2]; q[2]=p3[0]
        dq=qxyz(p3[1],p3[2],q0[5]); qw=qmul(dq,q0[3:7]); q[3:7]=qw/np.linalg.norm(qw); return q
    def model_at(qleg,p3):
        q=build(qleg,p3)
        c1=fwd(q,np.ones(90)); c0=fwd(q,np.zeros(90))
        g=np.asarray(c1['af'])-np.asarray(c0['af']); b=np.asarray(c0['af']); mom=c0['mom']
        Bm=mom.T*g[None,:]; base=mom.T@b+np.asarray(c0['pas'])-np.asarray(c0['bias'])
        types=c0['efc_type']; f0=c0['efc_D']*c0['efc_aref']; f0=np.where(types==0,f0,np.maximum(f0,0.))
        return c0,Bm,base,f0
    def geo(c,f0):
        dep=0.;nf=0.;bear=set();mar=[]
        for ci,(g1,g2) in enumerate(c['contact_geom']):
            g1=int(g1);g2=int(g2);leg=gleg.get(g2 if g1 in floors else g1) if ((g1 in floors)!=(g2 in floors)) else None
            pen=max(0.,-float(c['cdist'][ci])); addr=int(c['contact_efc_address'][ci]); dim=int(c['contact_dim'][ci])
            if leg is None: nf=max(nf,pen)
            else: dep=max(dep,pen)
            if addr<0: continue
            cnt=int(c['contact_row_count'][ci])
            if dim==1: loc=np.r_[f0[addr],0.,0.]
            else:
                e=f0[addr:addr+4]; mu=c['cfric'][ci,:2]; loc=np.r_[e.sum(),mu[0]*(e[0]-e[1]),mu[1]*(e[2]-e[3])]
            mar.append(float(loc[0]-sum(abs(loc[1:]/mu))))
            if leg is not None and loc[0]>0 and c['cdist'][ci]<=0: bear.add(leg)
        return dict(foot_depth=float(dep),nonfoot_depth=float(nf),bearing_legs=len(bear),friction_margin=float(min(mar,default=0.)))
    rows=[];state=dict(best=None)
    def evaluate(qleg,p3,tag):
        key=qleg.tobytes()+p3.tobytes()
        if key in cache:
            calls['cache_hits']+=1; return cache[key]
        c0,Bm,base,f0=model_at(qleg,p3)
        A=(Bm@T)[6:,:]; req=-(base+c0['efc_J_dense'].T@f0)[6:]
        i=TARGET_DOF-6
        lo=float(np.minimum(A[i],0).sum()); hi=float(np.maximum(A[i],0).sum())
        viol=float(max(0.,req[i]-hi,lo-req[i]))
        mneed=(float(req[i])/hi) if (req[i]>0 and hi>1e-12) else ((float(req[i])/lo) if (req[i]<0 and lo<-1e-12) else None)
        g=geo(c0,f0); gok=bool(g['foot_depth']<=DEPTH_CAP and g['nonfoot_depth']<=1e-12 and g['bearing_legs']>=3 and g['friction_margin']>=-1e-10)
        r=dict(tag=tag,dof22_required=float(req[i]),dof22_lo=lo,dof22_hi=hi,dof22_violation=viol,
               dof22_needed_multiplier=mneed,geometry=g,geom_ok=gok,qleg=qleg.tolist(),p3=p3.tolist())
        cache[key]=r; calls['candidate_evaluations']+=1; rows.append(r)
        b=state['best']
        if b is None or (r['dof22_violation'],0 if r['geom_ok'] else 1) < (b['dof22_violation'],0 if b['geom_ok'] else 1):
            state['best']=r
        return r
    dump(out/'PREREG.json',dict(task='S187/P1-S2 targeted attack on DOF 22 (LMCoxa_roll), the only row with m=1.32',
        objective='that single row interval violation; the 42-row LP is NOT the search objective',
        constraints='geometry gates hard; root held at the S177 root-balanced point during the search, then re-checked with a dim-3 root Newton',
        variables='LM leg 7 joints within native XML limits', budget=dict(forward_cap=CAP,reserved=RESERVED),
        context='DOF 42 is recorded as structurally unsatisfiable under the frozen interface (its only positive actuator is one of the 16 unmapped slots); this does NOT relax any acceptance criterion',
        no_step=True,no_neural=True,no_training=True,no_dynamics=True,units='native; SI NOT_ESTABLISHED'))
    qleg0=st['qpos'][7:].copy(); p3_0=np.array([p177[2],p177[3],p177[4]])
    lm=[k for k in range(42) if jn[k][6:8]=='LM']
    evaluate(qleg0,p3_0,'BASELINE')
    cur=qleg0.copy(); cur_v=state['best']['dof22_violation']; step=.15; sweeps=0; moves=0
    while step>=.02 and calls['forward']<CAP-RESERVED:
        sweeps+=1; improved=False
        for k in lm:
            for sgn in (+1,-1):
                if calls['forward']>=CAP-RESERVED: break
                cand=cur.copy(); cand[k]=float(np.clip(cand[k]+sgn*step*(nat[k][1]-nat[k][0]),nat[k][0],nat[k][1]))
                r=evaluate(cand,p3_0,f'lm_j{k}_{sgn:+d}_{step:g}')
                if r['geom_ok'] and r['dof22_violation']<cur_v-1e-12:
                    cur=cand; cur_v=r['dof22_violation']; improved=True; moves+=1
            if calls['forward']>=CAP-RESERVED: break
        if not improved: step/=2
    dump(out/'CANDIDATES.json',sorted(rows,key=lambda r:(r['dof22_violation'],0 if r['geom_ok'] else 1))[:80])
    best=state['best']
    # full check at the best point: dim-3 root Newton + 42-row LP
    check=None
    if best is not None:
        qleg=np.array(best['qleg'],float)
        def cb(p3):
            if calls['forward']>=CAP-6: raise RuntimeError('budget')
            c0,Bm,base,f0=model_at(qleg,p3)
            return dict(residual6=-(base+c0['efc_J_dense'].T@f0)[:6])
        lo3=np.array([p177[2]*.85,-.35,-.35]); hi3=np.array([p177[2]*1.15,.35,.35])
        nr=RootNewton(cb,lo3,hi3,cap=10**6,max_iter=6,tol=1e-8,name='c3')
        try:
            res=nr.solve(p3_0); rn=float(res['norm_inf']); p3=np.array(res['p'])
        except Exception: rn=float('inf'); p3=p3_0
        calls['root_newton_iterations']+=nr.ledger['newton_iterations']; calls['root_jacobians']+=nr.ledger['jacobians']
        c0,Bm,base,f0=model_at(qleg,p3); A=(Bm@T)[6:,:]; req=-(base+c0['efc_J_dense'].T@f0)[6:]
        n=A.shape[1]
        lp=linprog(np.r_[np.zeros(n),1.],A_ub=np.vstack((np.column_stack((A,-np.ones(42))),np.column_stack((-A,-np.ones(42))))),
                   b_ub=np.r_[req,-req],bounds=[(0,1)]*n+[(0,None)],method='highs'); calls['inner_lp']+=1
        lo_r=np.minimum(A,0).sum(axis=1); hi_r=np.maximum(A,0).sum(axis=1)
        reach=(lo_r-1e-9<=req)&(req<=hi_r+1e-9); g=geo(c0,f0)
        check=dict(root_norm=rn,root_ok=bool(rn<=1e-8),minimax=float(lp.x[-1]) if lp.success else None,
                   unreachable=[int(d) for d in range(6,48) if not reach[d-6]],geometry=g,
                   geom_ok=bool(g['foot_depth']<=DEPTH_CAP and g['nonfoot_depth']<=1e-12 and g['bearing_legs']>=3 and g['friction_margin']>=-1e-10),
                   dof22_violation=float(max(0.,req[TARGET_DOF-6]-np.maximum(A[TARGET_DOF-6],0).sum(),np.minimum(A[TARGET_DOF-6],0).sum()-req[TARGET_DOF-6])))
        dump(out/'BEST_FULL_CHECK.json',check)
        if check['minimax'] is not None and check['minimax']<=1e-8 and check['root_ok'] and check['geom_ok']:
            cf=fwd(np.array(build(qleg,p3)),T@lp.x[:n])
            dump(out/'CONFIRMATION.json',dict(max_abs_residual48=None,max_abs_qacc=float(np.abs(cf['qacc']).max()),
                 act_dot_all_zero=bool(not np.any(cf['ad']))))
    dump(out/'SUMMARY.json',dict(best_dof22_violation=None if best is None else best['dof22_violation'],
        best_geom_ok=None if best is None else best['geom_ok'], moves=moves, sweeps=sweeps, step=step,
        dof22_reached_zero=bool(best is not None and best['dof22_violation']<=1e-9),
        full_check=check, calls=calls,
        note='DOF 42 recorded separately as structurally unsatisfiable under the frozen interface; acceptance criteria unchanged',
        units='native; SI NOT_ESTABLISHED'))
    dump(out/'INPUT_HASHES_AFTER.json',{k:sha(k) for k in h})
    print(json.dumps(dict(best_viol=None if best is None else best['dof22_violation'], geom_ok=None if best is None else best['geom_ok'],
        reached=bool(best is not None and best['dof22_violation']<=1e-9), moves=moves, fw=calls['forward'],
        check_minimax=None if not check else check['minimax'])))
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path)
    a=ap.parse_args(); main(a.output or B_/'s187-dof22-targeted'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
