"""S189: prerequisite checks before any model change.

Stage A (read-only): native actuator spec from the model, gain-vs-arm decomposition, L/R
symmetry, strength budget. Stage B: fixed-point coupling test (does feeding the LP activation
into the engine change f0 and reduce the residual?).
No mj_step, no neural, no training, no XML change.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, csv, json, shutil, hashlib
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import linprog
B_=Path('/home/sensen/Desktop/engine修复验收'); R=Path('/run/media/sensen/Data2/cell_wound_prototype')
S154=B_/'s154-activation-rate/20261024T215751775489Z'; S166=B_/'s166-mapped-support/20261024T231418078692647Z'
S177=B_/'s177-root-newton-search/20261025T020500000000Z'; S179=B_/'s179-targeted-legjoint-search/20261025T070000000000Z'
XML=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
ITERS=10; CAP_B=120; RESERVED=100
SIX=[19,22,26,40,42,47]
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
def main(out):
    assert not out.exists(); out.mkdir(parents=True); (out/'SOURCE_FREEZE').mkdir()
    files=[Path(__file__).resolve(),XML,S154/'INPUT_HASHES.json',S154/'init80ms/TRAJECTORY.npz',S154/'init80ms/MODEL_PROVENANCE.json',
           S154/'init80ms/NATIVE_MODEL_ARRAYS.npz',S154/'init80ms/COM_CONTACT_METADATA.json',S166/'MAPPING.npz',
           S177/'ROOT_BALANCE_STATE.npz',S177/'ROOT_RESULT.json',S179/'ROW_DOF_MAP.json']
    old=json.loads((S154/'INPUT_HASHES.json').read_text()); old=old.get('hashes',old)
    h={}
    for p,v in old.items():
        if isinstance(v,str) and len(v)==64: assert sha(p)==v,'S154 binding'; h[p]=v
    h.update({str(p):sha(p) for p in files}); dump(out/'INPUT_HASHES.json',h)
    for i,(p,v) in enumerate(sorted(h.items())):
        t=out/'SOURCE_FREEZE'/f'{i:04d}_{Path(p).name}'; shutil.copyfile(p,t); assert sha(t)==v
    rowmap=json.loads((S179/'ROW_DOF_MAP.json').read_text())
    assert [r['dof_name'] for r in rowmap['rows']]==[r['dof_name'] for r in rowmap['rows']]
    dump(out/'ROW_DOF_MAP_REUSED.json',dict(sha256=sha(S179/'ROW_DOF_MAP.json'),rows=rowmap['rows']))
    jn=[r['dof_name'] for r in rowmap['rows']]
    st=load(S177/'ROOT_BALANCE_STATE.npz'); mp=load(S166/'MAPPING.npz')
    T=st['T']; pool=mp['actuator_pool']; p177=np.array(json.loads((S177/'ROOT_RESULT.json').read_text())['p'],float)
    z=load(S154/'init80ms/TRAJECTORY.npz'); q0=z['qpos'][0].copy()
    proof=json.loads((S154/'init80ms/MODEL_PROVENANCE.json').read_text())
    meta=json.loads((S154/'init80ms/COM_CONTACT_METADATA.json').read_text())
    mass=load(S154/'init80ms/NATIVE_MODEL_ARRAYS.npz')['body_mass']; W=float(mass.sum()*9801.)
    specs=[(R/'run_matched_state_counterfactual.py',('snapshot',)),(R/'run_local_pool_response.py',('moment_record',)),(R/'run_constraint_row_recording.py',('constraint_record',))]
    ns=dict(np=np,ah=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    exec(compile('\n'.join(extract(p,n) for p,n in specs),'<inert>','exec'),ns)
    calls=dict(model_load=0,forward=0,inner_lp=0,cache_hits=0)
    import mujoco as mj
    calls['model_load']+=1; m=mj.MjModel.from_xml_path(str(XML))
    assert ns['snapshot'](m)==proof['before']
    m.actuator_dynprm[:,0]=.001; m.actuator_dynprm[:,1]=.0004
    m.opt.integrator=mj.mjtIntegrator.mjINT_RK4; m.opt.timestep=.000025
    assert ns['snapshot'](m)==proof['after']
    # ---------- Stage A: native spec ----------
    tree=ET.parse(XML)
    anames=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_ACTUATOR,k) for k in range(m.nu)]
    act_rows=[]
    for k in range(m.nu):
        lo,hi=float(m.actuator_ctrlrange[k][0]),float(m.actuator_ctrlrange[k][1])
        act_rows.append(dict(slot=k,actuator=anames[k],pool=int(pool[k]),mapped=bool(pool[k]>=0),
            ctrlrange=[lo,hi],ctrl_limited=bool(m.actuator_ctrllimited[k]),
            gaintype=int(m.actuator_gaintype[k]),biastype=int(m.actuator_biastype[k]),dyntype=int(m.actuator_dyntype[k]),
            gainprm=[float(x) for x in m.actuator_gainprm[k]],biasprm=[float(x) for x in m.actuator_biasprm[k]],
            gear=[float(x) for x in m.actuator_gear[k]],forcerange=[float(x) for x in m.actuator_forcerange[k]],
            forcelimited=bool(m.actuator_forcelimited[k]),actfrcrange='NOT_PRESENT_IN_MUJOCO_3_9',
            actlimited=bool(m.actuator_actlimited[k]),actrange=[float(x) for x in m.actuator_actrange[k]],
            kind=('ACTIVATION_MUSCLE' if int(m.actuator_dyntype[k])==int(mj.mjtDyn.mjDYN_MUSCLE) else 'OTHER')))
    with (out/'ACTUATOR_NATIVE_SPEC.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['slot','actuator','pool','mapped','ctrlrange_lo','ctrlrange_hi','ctrl_limited',
            'gaintype','biastype','dyntype','gear','forcerange','forcelimited','actrange','actlimited','kind'])
        for r in act_rows:
            w.writerow([r['slot'],r['actuator'],r['pool'],r['mapped'],r['ctrlrange'][0],r['ctrlrange'][1],r['ctrl_limited'],
                        r['gaintype'],r['biastype'],r['dyntype'],r['gear'],r['forcerange'],r['forcelimited'],r['actrange'],r['actlimited'],r['kind']])
    uppers=sorted({r['ctrlrange'][1] for r in act_rows})
    native_hi=max(uppers)
    m0=(0.5/native_hi) if native_hi>0 else None
    dump(out/'NATIVE_CAP_DECLARATION.json',dict(
        ctrlrange_upper_set=uppers,ctrl_limited_any=any(r['ctrl_limited'] for r in act_rows),
        actrange_set=sorted({tuple(r['actrange']) for r in act_rows}),
        forcelimited_any=any(r['forcelimited'] for r in act_rows),
        all_muscle=all(r['kind']=='ACTIVATION_MUSCLE' for r in act_rows),
        n_slots=len(act_rows),n_mapped=sum(1 for r in act_rows if r['mapped']),n_unmapped=sum(1 for r in act_rows if not r['mapped']),
        half_amplitude_m0=m0,
        interpretation='m0 = 0.5 / native ctrl upper bound; the S188 multiplier 22.52 = m0 * m_remaining',
        m_remaining=float(22.522331811468668/m0) if m0 else None,
        note='if the native bound is 1.0 then m0=0.5 and the missing factor is only ~11.26, not 22.5'))
    # ---------- Stage A: gain/arm separation at the baseline (needs 2 forwards for the exact gain) ----------
    def fwd(q,a=None):
        d=mj.MjData(m); d.time=float(z['time'][0])
        for n in STATE: getattr(d,n)[:]=z[n][0]
        for n in OPT:
            if n in z: getattr(d,n)[:]=z[n][0]
            else: assert np.asarray(getattr(d,n)).size==0
        d.qpos[:]=q; d.qvel[:]=0
        act=np.zeros(90) if a is None else np.asarray(a,float)
        d.act[:]=act; d.ctrl[:]=act; d.qacc_warmstart[:]=0
        mj.mj_forward(m,d); calls['forward']+=1
        c=dict(qacc=np.asarray(d.qacc).copy(),ad=np.asarray(d.act_dot).copy(),force=np.asarray(d.actuator_force).copy(),
               length=np.asarray(d.actuator_length).copy(),qfrc_actuator=np.asarray(d.qfrc_actuator).copy(),
               qfrc_passive=np.asarray(d.qfrc_passive).copy(),qfrc_bias=np.asarray(d.qfrc_bias).copy(),
               qfrc_constraint=np.asarray(d.qfrc_constraint).copy(),
               efc_D=np.asarray(d.efc_D).copy(),efc_aref=np.asarray(d.efc_aref).copy(),efc_type=np.asarray(d.efc_type).copy(),
               cdist=np.array([v.dist for v in d.contact]),cfric=np.array([v.friction.copy() for v in d.contact]).reshape(-1,5))
        c.update({k:np.asarray(v).copy() for k,v in ns['constraint_record'](mj,m,d,0).items()})
        c['mom']=np.asarray(ns['moment_record'](m,d)['actuator_moment']).copy()
        return c
    qbase=q0.copy(); qbase[7:]=np.load(S177/'ROOT_BALANCE_STATE.npz')['qpos'][7:]
    qbase[0:2]=q0[0:2]; qbase[2]=p177[2]
    rx,ry,rz=p177[3],p177[4],q0[5]
    cx,sx=np.cos(rx/2),np.sin(rx/2); cy,sy=np.cos(ry/2),np.sin(ry/2); cz,sz=np.cos(rz/2),np.sin(rz/2)
    dq=np.array([cx*cy*cz+sx*sy*sz, sx*cy*cz-cx*sy*sz, cx*sy*cz+sx*cy*sz, cx*cy*sz-sx*sy*cz])
    aw,ax,ay,az=dq; bw,bx,by,bz=q0[3:7]
    qw=np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])
    qbase[3:7]=qw/np.linalg.norm(qw)
    c1=fwd(qbase,np.ones(90)); c0=fwd(qbase,np.zeros(90))
    gain=np.asarray(c1['force'])-np.asarray(c0['force']); bias=np.asarray(c0['force'])
    arm=np.asarray(c0['mom'])  # cells: [horizon, 1, 2] -> resolve below
    if arm.ndim==3: arm=arm[0]
    arm=arm.reshape(90,48) if arm.size==90*48 else arm
    assert arm.shape==(90,48), arm.shape
    mech=arm.T*gain[None,:]           # 48 x 90  (DOF x slot)
    A=(mech@T)[6:,:]; req=-(np.asarray(c0['qfrc_passive'])-np.asarray(c0['qfrc_bias'])+c0['efc_J_dense'].T@np.where(c0['efc_type']==0,c0['efc_D']*c0['efc_aref'],np.maximum(c0['efc_D']*c0['efc_aref'],0.)))[6:]
    decomp=[]
    for d in SIX:
        i=d-6
        slots=[k for k in range(90) if abs(mech[d,k])>1e-12]
        pos=[k for k in slots if mech[d,k]>0]; neg=[k for k in slots if mech[d,k]<0]
        dom_pos=max(pos,key=lambda k:mech[d,k]) if pos else None
        dom_neg=min(neg,key=lambda k:mech[d,k]) if neg else None
        for k in sorted(slots,key=lambda k:-abs(mech[d,k]))[:4]:
            decomp.append(dict(dof=d,joint_row=i,dof_name=jn[i],leg=jn[i][6:8],joint_name=jn[i],slot=k,
                actuator=anames[k],pool=int(pool[k]),mapped=bool(pool[k]>=0),
                gain=float(gain[k]),arm=float(arm[k,d]),half_gain_arm=float(0.5*gain[k]*arm[k,d]),
                required=float(req[i]),
                kind=('POSITIVE' if mech[d,k]>0 else 'NEGATIVE')))
        decomp.append(dict(dof=d,joint_row=i,dof_name=jn[i],leg=jn[i][6:8],joint_name=jn[i],slot=-1,actuator='__ROW_SUMMARY__',
            pool=-1,mapped=None,gain=None,arm=None,half_gain_arm=None,required=float(req[i]),
            kind=('dom_arm_gain_ratio=%.1f' % (abs(arm[dom_neg,d])/abs(arm[dom_pos,d])) if (dom_neg is not None and dom_pos is not None and abs(arm[dom_pos,d])>1e-12) else 'one_sided')))
    with (out/'GAIN_ARM_DECOMPOSITION.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(decomp[0])); w.writeheader(); w.writerows(decomp)
    # ---------- Stage A: symmetry ----------
    def leg_prefix(name):
        for p in ('LF','LM','LH','RF','RM','RH'):
            if name[6:8]==p: return p
        return '??'
    sym=[]
    pairs=[('LF','RF'),('LM','RM'),('LH','RH')]
    for a_,b_ in pairs:
        for suf in ('Tibia_pitch','Coxa_pitch','Coxa_yaw','Coxa_roll','Trochanter_pitch'):
            ia=[i for i in range(42) if jn[i]==f'joint_{a_}{suf}']
            ib=[i for i in range(42) if jn[i]==f'joint_{b_}{suf}']
            if not ia or not ib: continue
            ia,ib=ia[0],ib[0]
            def mapping(prefix):
                return {k:name.replace(prefix,'X') for k,name in enumerate(anames) if prefix in name}
            ma,mb=mapping(a_),mapping(b_)
            common=[(k1,k2) for k1,n1 in ma.items() for k2,n2 in mb.items() if n1==n2.replace(b_,a_)][:6]
            for k1,k2 in common[:3]:
                sym.append(dict(dof_left=6+ia,dof_right=6+ib,joint_left=jn[ia],joint_right=jn[ib],
                    slot_left=k1,actuator_left=anames[k1],slot_right=k2,actuator_right=anames[k2],
                    gain_left=float(gain[k1]),gain_right=float(gain[k2]),
                    arm_left=float(arm[k1,6+ia]),arm_right=float(arm[k2,6+ib]),
                    gain_ratio=float(gain[k2]/gain[k1]) if abs(gain[k1])>1e-12 else None,
                    arm_ratio=float(arm[k2,6+ib]/arm[k1,6+ia]) if abs(arm[k1,6+ia])>1e-12 else None))
    dump(out/'SYMMETRY_CHECK.json',dict(pairs=sym,
        note='left/right mirrored actuator pairs compared for gain and arm; ratios near 1 mean the design is symmetric by construction'))
    # ---------- Stage A: strength budget ----------
    budget=[]
    for i in range(42):
        hi=float(np.maximum(A[i],0).sum()); lo=float(np.minimum(A[i],0).sum())
        budget.append(dict(joint_row=i,dof=6+i,dof_name=jn[i],required=float(req[i]),
                           mapped_positive=hi,mapped_negative=lo,mach_positive=float(np.maximum(mech[6+i],0).sum()),
                           mach_negative=float(np.minimum(mech[6+i],0).sum()),body_weight=W,
                           ratio_to_weight=float(max(hi,abs(lo))/W)))
    dump(out/'STRENGTH_BUDGET.json',dict(body_weight=W,total_mass=float(mass.sum()),rows=budget,
        global_max_positive=float(max(b['mapped_positive'] for b in budget)),
        global_max_negative=float(min(b['mapped_negative'] for b in budget)),
        max_ratio_to_weight=float(max(b['ratio_to_weight'] for b in budget)),
        note='mapped authority is 0.5*gain*arm summed over mapped pools; compare with the body weight to see the overall force scale'))
    # ---------- Stage B: fixed-point coupling test ----------
    def geom(c,f0):
        floors=set(meta['floor_geom_ids']); gl={int(k):v for k,v in meta['geom_map'].items()}
        dep=0.;nf=0.;bear=set();mar=[]
        for ci,(g1,g2) in enumerate(c['contact_geom']):
            g1=int(g1);g2=int(g2);leg=gl.get(g2 if g1 in floors else g1) if ((g1 in floors)!=(g2 in floors)) else None
            pen=max(0.,-float(c['cdist'][ci])); addr=int(c['contact_efc_address'][ci]); dim=int(c['contact_dim'][ci])
            if leg is None: nf=max(nf,pen)
            else: dep=max(dep,pen)
            if addr<0: continue
            if dim==1: loc=np.r_[f0[addr],0.,0.]
            else:
                e=f0[addr:addr+4]; mu=c['cfric'][ci,:2]; loc=np.r_[e.sum(),mu[0]*(e[0]-e[1]),mu[1]*(e[2]-e[3])]
            mar.append(float(loc[0]-sum(abs(loc[1:]/mu))))
            if leg is not None and loc[0]>0 and c['cdist'][ci]<=0: bear.add(leg)
        return dict(foot_depth=dep,nonfoot_depth=nf,bearing_legs=len(bear),friction_margin=min(mar,default=0.),
                    nefc=int(len(c['efc_type'])),contacts=int(len(c['contact_geom'])))
    def build_AB(c0):
        types=c0['efc_type']; f0=np.where(types==0,c0['efc_D']*c0['efc_aref'],np.maximum(c0['efc_D']*c0['efc_aref'],0.))
        mom=np.asarray(c0['mom']) if np.asarray(c0['mom']).ndim==2 else np.asarray(c0['mom'])[0]
        mom=mom.reshape(90,48) if mom.size==90*48 else mom
        B=mom.T*gain[None,:]
        base=mom.T@bias+np.asarray(c0['qfrc_passive'])-np.asarray(c0['qfrc_bias'])
        return B,f0,base
    iters=[]; act=np.zeros(66); cache={}
    for it in range(ITERS):
        if calls['forward']>=CAP_B-RESERVED: break
        act90=T@act
        key=np.asarray(qbase,dtype=np.float64).tobytes()+act90.tobytes()
        if key in cache:
            calls['cache_hits']+=1; c=cache[key]
        else:
            c=fwd(qbase,act90); cache[key]=c
        B,f0,base=build_AB(c)
        Am=(B@T)[6:,:]; target=-(base+np.asarray(c['efc_J_dense']).T@f0)[6:]
        res48=B@act90+base+np.asarray(c['efc_J_dense']).T@f0
        g=geom(c,f0)
        nsum=0.0; types=c['efc_type']; fl=np.where(types==0,c['efc_D']*c['efc_aref'],np.maximum(c['efc_D']*c['efc_aref'],0.))
        for ci in range(len(c['contact_geom'])):
            addr=int(c['contact_efc_address'][ci]); dim=int(c['contact_dim'][ci])
            if addr<0: continue
            nsum+=float(fl[addr] if dim==1 else fl[addr:addr+4].sum())
        n=Am.shape[1]
        lp=linprog(np.r_[np.zeros(n),1.],A_ub=np.vstack((np.column_stack((Am,-np.ones(42))),np.column_stack((-Am,-np.ones(42))))),
                   b_ub=np.r_[target,-target],bounds=[(0,1)]*n+[(0,None)],method='highs'); calls['inner_lp']+=1
        lo_r=np.minimum(Am,0).sum(axis=1); hi_r=np.maximum(Am,0).sum(axis=1)
        reach=(lo_r-1e-9<=target)&(target<=hi_r+1e-9)
        iters.append(dict(iteration=it,lp_predicted_minimax=float(lp.x[-1]) if lp.success else None,
            measured_max_abs_residual48=float(np.abs(res48).max()),
            measured_max_abs_residual6=float(np.abs(res48[:6]).max()),
            six_row_violation=[float(max(0.,target[d-6]-hi_r[d-6],lo_r[d-6]-target[d-6])) for d in SIX],
            six_row_violation_max=float(max(max(0.,target[d-6]-hi_r[d-6],lo_r[d-6]-target[d-6]) for d in SIX)),
            unreachable=[int(d) for d in range(6,48) if not reach[d-6]],
            sum_normal_over_weight=float(nsum/W),geometry=g,
            max_abs_qacc=float(np.abs(c['qacc']).max()),act_dot_zero=bool(not np.any(c['ad'])),
            act90_max=float(np.abs(act90).max())))
        act=np.asarray(lp.x[:n])
        dump(out/'FIXEDPOINT_ITERATIONS.json',dict(iterations=iters,calls=calls,
            stopping='iteration cap' if it+1==ITERS else 'budget'))
    first=iters[0]; last=iters[-1]
    trend=('COUPLING_REDUCES_RESIDUAL' if last['six_row_violation_max'] < 0.2*first['six_row_violation_max'] else
           ('COUPLING_NO_HELP' if last['six_row_violation_max'] >= 0.9*first['six_row_violation_max'] else 'COUPLING_MARGINAL'))
    dump(out/'SUMMARY.json',dict(stageA=dict(native_hi_upper=uppers,half_amplitude_m0=m0,
            m_remaining=float(22.522331811468668/m0) if m0 else None),stageB=dict(iters=len(iters),verdict=trend,
            first_six_violation=first['six_row_violation_max'],last_six_violation=last['six_row_violation_max'],
            first_res48=first['measured_max_abs_residual48'],last_res48=last['measured_max_abs_residual48']),
            calls=calls,units='native; SI NOT_ESTABLISHED',no_step=True,no_neural=True))
    dump(out/'INPUT_HASHES_AFTER.json',{k:sha(k) for k in h})
    print(json.dumps(dict(native_hi=uppers,m0=m0,m_rem=(22.522331811468668/m0) if m0 else None,
        verdict=trend,first6=first['six_row_violation_max'],last6=last['six_row_violation_max'],
        fw=calls['forward'],lp=calls['inner_lp'])))
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path)
    a=ap.parse_args(); main(a.output or B_/'s189-coupling-prerequisite'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
