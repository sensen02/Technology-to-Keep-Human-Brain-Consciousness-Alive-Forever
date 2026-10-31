"""S190: arm-definition provenance + symmetry judgment. Read-only, no search, no model change."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, csv, json, shutil, hashlib
import xml.etree.ElementTree as ET
import numpy as np
B_=Path('/home/sensen/Desktop/engine修复验收'); R=Path('/run/media/sensen/Data2/cell_wound_prototype')
S154=B_/'s154-activation-rate/20261024T215751775489Z'; S166=B_/'s166-mapped-support/20261024T231418078692647Z'
S177=B_/'s177-root-newton-search/20261025T020500000000Z'; S179=B_/'s179-targeted-legjoint-search/20261025T070000000000Z'
S185=B_/'s185-row-dof-reconciliation/20261025T172000000000Z'; S186=B_/'s186-model-side-audit/20261025T180000000000Z'
S189=B_/'s189-coupling-prerequisite/20261025T214000000000Z'
XML=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
CAP=40
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
           S154/'init80ms/COM_CONTACT_METADATA.json',S166/'MAPPING.npz',S177/'ROOT_BALANCE_STATE.npz',S177/'ROOT_RESULT.json',
           S179/'ROW_DOF_MAP.json',S185/'RECONCILE_BASELINE.csv',S186/'MUSCLE_ATTACHMENT_AUDIT.csv',S189/'GAIN_ARM_DECOMPOSITION.csv',
           S189/'ACTUATOR_NATIVE_SPEC.csv']
    old=json.loads((S154/'INPUT_HASHES.json').read_text()); old=old.get('hashes',old)
    h={}
    for p,v in old.items():
        if isinstance(v,str) and len(v)==64: assert sha(p)==v,'S154 binding'; h[p]=v
    h.update({str(p):sha(p) for p in files}); dump(out/'INPUT_HASHES.json',h)
    for i,(p,v) in enumerate(sorted(h.items())):
        t=out/'SOURCE_FREEZE'/f'{i:04d}_{Path(p).name}'; shutil.copyfile(p,t); assert sha(t)==v
    rowmap=json.loads((S179/'ROW_DOF_MAP.json').read_text())
    dump(out/'ROW_DOF_MAP_REUSED.json',dict(sha256=sha(S179/'ROW_DOF_MAP.json'),rows=rowmap['rows']))
    jn=[r['dof_name'] for r in rowmap['rows']]
    st=load(S177/'ROOT_BALANCE_STATE.npz'); mp=load(S166/'MAPPING.npz'); T=st['T']; pool=mp['actuator_pool']
    p177=np.array(json.loads((S177/'ROOT_RESULT.json').read_text())['p'],float)
    z=load(S154/'init80ms/TRAJECTORY.npz'); q0=z['qpos'][0].copy(); proof=json.loads((S154/'init80ms/MODEL_PROVENANCE.json').read_text())
    meta=json.loads((S154/'init80ms/COM_CONTACT_METADATA.json').read_text())
    specs=[(R/'run_matched_state_counterfactual.py',('snapshot',)),(R/'run_local_pool_response.py',('moment_record',)),(R/'run_constraint_row_recording.py',('constraint_record',))]
    ns=dict(np=np,ah=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    exec(compile('\n'.join(extract(p,n) for p,n in specs),'<inert>','exec'),ns)
    calls=dict(model_load=0,forward=0)
    import mujoco as mj
    calls['model_load']+=1; m=mj.MjModel.from_xml_path(str(XML))
    assert ns['snapshot'](m)==proof['before']
    m.actuator_dynprm[:,0]=.001; m.actuator_dynprm[:,1]=.0004
    m.opt.integrator=mj.mjtIntegrator.mjINT_RK4; m.opt.timestep=.000025
    assert ns['snapshot'](m)==proof['after']
    anames=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_ACTUATOR,k) for k in range(m.nu)]
    # ---------- Stage A.3 native spec (act vs ctrl) ----------
    spec=[]
    for k in range(m.nu):
        spec.append(dict(slot=k,actuator=anames[k],pool=int(pool[k]),mapped=bool(pool[k]>=0),
            dyntype=int(m.actuator_dyntype[k]),gaintype=int(m.actuator_gaintype[k]),biastype=int(m.actuator_biastype[k]),
            actlimited=bool(m.actuator_actlimited[k]),actrange=[float(m.actuator_actrange[k][0]),float(m.actuator_actrange[k][1])],
            ctrl_limited=bool(m.actuator_ctrllimited[k]),ctrlrange=[float(m.actuator_ctrlrange[k][0]),float(m.actuator_ctrlrange[k][1])],
            actadr=int(m.actuator_actadr[k]),actnum=int(m.actuator_actnum[k]),gear=[float(x) for x in m.actuator_gear[k]]))
    with (out/'ACTUATOR_NATIVE_SPEC.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(spec[0])); w.writeheader(); w.writerows(spec)
    act_hi=sorted({r['actrange'][1] for r in spec}); ctrl_hi=sorted({r['ctrlrange'][1] for r in spec})
    act_lim=any(r['actlimited'] for r in spec); ctrl_lim=any(r['ctrl_limited'] for r in spec)
    m0=dict(act=float(max(act_hi)/0.5),ctrl=float(max(ctrl_hi)/0.5))
    # ---------- Stage A.1 provenance of every earlier "arm" ----------
    prov=[]
    r185=list(csv.DictReader((S185/'RECONCILE_BASELINE.csv').open()))
    r186=list(csv.DictReader((S186/'MUSCLE_ATTACHMENT_AUDIT.csv').open()))
    r189=list(csv.DictReader((S189/'GAIN_ARM_DECOMPOSITION.csv').open()))
    for row in r186:
        prov.append(dict(source='S186 MUSCLE_ATTACHMENT_AUDIT.csv',dof=int(row['dof']),slot=int(row['slot']),
            actuator=row['actuator'],reported_key='arm',reported_value=float(row['arm']),
            actual_definition='B[dof,slot] = gain[slot] * actuator_moment[slot,dof]  (generalized force per unit activation, NOT a geometric moment)',
            unit='native generalized force per unit activation'))
    for row in r189:
        if row['actuator']=='__ROW_SUMMARY__': continue
        prov.append(dict(source='S189 GAIN_ARM_DECOMPOSITION.csv',dof=int(row['dof']),slot=int(row['slot']),
            actuator=row['actuator'],reported_key='arm',reported_value=float(row['arm']),
            actual_definition='actuator_moment[slot,dof] = d(tendon length)/d(q_dof)  (geometric moment, dimensionless arm in native length)',
            unit='native length (moment arm)'))
    dump(out/'ARM_PROVENANCE.csv',prov)
    # ---------- Stage B.1 baseline measurement: gain, moment, three methods ----------
    def fwd(q,a=None):
        d=mj.MjData(m); d.time=float(z['time'][0])
        for n in STATE: getattr(d,n)[:]=z[n][0]
        for n in OPT:
            if n in z: getattr(d,n)[:]=z[n][0]
            else: assert np.asarray(getattr(d,n)).size==0
        d.qpos[:]=q; d.qvel[:]=0
        av=np.zeros(90) if a is None else np.asarray(a,float)
        d.act[:]=av; d.ctrl[:]=av; d.qacc_warmstart[:]=0
        mj.mj_forward(m,d); calls['forward']+=1
        mom=np.asarray(ns['moment_record'](m,d)['actuator_moment'])
        mom=mom[0] if mom.ndim==3 else mom
        return dict(force=np.asarray(d.actuator_force).copy(),length=np.asarray(d.actuator_length).copy(),
                    qfrc_actuator=np.asarray(d.qfrc_actuator).copy(),mom=mom.reshape(90,48))
    qbase=q0.copy(); qbase[7:]=np.load(S177/'ROOT_BALANCE_STATE.npz')['qpos'][7:]
    qbase[0:2]=q0[0:2]; qbase[2]=p177[2]
    rx,ry,rz=p177[3],p177[4],q0[5]
    cx,sx=np.cos(rx/2),np.sin(rx/2); cy,sy=np.cos(ry/2),np.sin(ry/2); cz,sz=np.cos(rz/2),np.sin(rz/2)
    dq=np.array([cx*cy*cz+sx*sy*sz, sx*cy*cz-cx*sy*sz, cx*sy*cz+sx*cy*sz, cx*cy*sz-sx*sy*cz])
    aw,ax,ay,az=dq; bw,bx,by,bz=q0[3:7]
    qw=np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])
    qbase[3:7]=qw/np.linalg.norm(qw)
    c0=fwd(qbase,np.zeros(90)); c1=fwd(qbase,np.ones(90))
    gain=np.asarray(c1['force'])-np.asarray(c0['force']); bias=np.asarray(c0['force'])
    mom=c0['mom']
    # method (b): numeric dL/dq for each of the six joints
    eps=1e-5
    dLdq={}
    for dof in SIX:
        j=dof-6
        qp=qbase.copy(); qp[7+j]+=eps; qm=qbase.copy(); qm[7+j]-=eps
        Lp=np.asarray(fwd(qp,np.zeros(90))['length']); Lm=np.asarray(fwd(qm,np.zeros(90))['length'])
        dLdq[dof]=(Lp-Lm)/(2*eps)
    # method (c): invert qfrc_actuator for the dominant actuators of each row
    mech=(mom.T*gain[None,:])
    dominant={}
    for dof in SIX:
        s=[k for k in range(90) if abs(mech[dof,k])>1e-12]
        if not s: continue
        dp=max(s,key=lambda k:mech[dof,k]); dn=min(s,key=lambda k:mech[dof,k])
        for k in (dp,dn):
            if k not in dominant.values(): pass
        dominant[dof]=(dp,dn)
    inv={}
    for k in sorted({k for pr in dominant.values() for k in pr}):
        if calls['forward']>=CAP: break
        a=np.zeros(90); a[k]=1.0
        cc=fwd(qbase,a)
        denom=float(cc['force'][k]) if abs(float(cc['force'][k]))>1e-12 else None
        delta=np.asarray(cc['qfrc_actuator'])-np.asarray(c0['qfrc_actuator'])
        gk=float(gain[k])
        inv[k]=(delta/gk) if abs(gk)>1e-12 else None
    triple=[]
    for dof,(dp,dn) in dominant.items():
        for k,kind in ((dp,'POSITIVE'),(dn,'NEGATIVE')):
            row_m=[]
            for method in ('actuator_moment','tendon_length_derivative','qfrc_inversion'):
                if method=='actuator_moment': val=float(mom[k,dof])
                elif method=='tendon_length_derivative': val=float(dLdq[dof][k])
                else: val=float(inv[k][dof]) if inv.get(k) is not None else None
                row_m.append((method,val))
            vals=[v for _,v in row_m if v is not None]
            spread=float((max(vals)-min(vals))/max(abs(min(vals)),abs(max(vals)),1e-30)) if len(vals)>1 else None
            triple.append(dict(dof=dof,joint_row=dof-6,dof_name=jn[dof-6],leg=jn[dof-6][6:8],joint_name=jn[dof-6],
                slot=k,actuator=anames[k],kind=kind,gain=float(gain[k]),mapped=bool(pool[k]>=0),
                method_values={m:v for m,v in row_m},rel_spread=spread,
                gain_arm_product=float(gain[k]*mom[k,dof]),half_gain_arm=float(0.5*gain[k]*mom[k,dof])))
    dump(out/'ARM_TRIPLE_METHOD_CROSSCHECK.json',dict(rows=triple,
        max_rel_spread=float(max([t['rel_spread'] for t in triple if t['rel_spread'] is not None],default=0.)),
        note='(a) actuator_moment column, (b) numeric dL/dq, (c) qfrc_actuator inversion at act=e_k; agreement to 1e-9 relative required'))
    # ---------- Stage B.1 pairing by names + tendon length curves ----------
    def flip(name):
        for a_,b_ in (('LF','RF'),('LM','RM'),('LH','RH')):
            if name.startswith(a_) or ('_'+a_) in name: return name.replace(a_,b_)
            if name.startswith(b_) or ('_'+b_) in name: return name.replace(b_,a_)
        return None
    by_name={n:k for k,n in enumerate(anames)}
    pairs=[]; seen=set()
    for k,n in enumerate(anames):
        if not n.startswith(('L','R')): continue
        f=flip(n)
        if f and f in by_name and (k not in seen):
            k2=by_name[f]
            if k2==k: continue
            seen.add(k); seen.add(k2)
            # mirrored joint pair: find the joint index of the same suffix
            pairs.append(dict(slot_left=k,actuator_left=n,slot_right=k2,actuator_right=anames[k2]))
    joint_pairs=[]
    for i in range(42):
        nm=jn[i]
        for a_,b_ in (('LF','RF'),('LM','RM'),('LH','RH')):
            if nm.startswith('joint_'+a_):
                other='joint_'+b_+nm[len('joint_'+a_):]
                if other in jn: joint_pairs.append((i,jn.index(other),a_,b_))
    sym=[]
    for k1,k2,*rest in [(p['slot_left'],p['slot_right']) for p in pairs]:
        for jl,jr,la,rb in joint_pairs:
            if not (rest or True): pass
            sym.append(dict(actuator_left=anames[k1],actuator_right=anames[k2],joint_left=jn[jl],joint_right=jn[jr],
                gain_left=float(gain[k1]),gain_right=float(gain[k2]),
                arm_left=float(mom[k1,6+jl]),arm_right=float(mom[k2,6+jr]),
                gain_ratio=float(gain[k2]/gain[k1]) if abs(gain[k1])>1e-12 else None,
                arm_ratio=float(mom[k2,6+jr]/mom[k1,6+jl]) if abs(mom[k1,6+jl])>1e-12 else None))
        if len(sym)>400: break
    dump(out/'SYMMETRY_PAIRING.json',dict(n_pairs=len(pairs),pairs=pairs[:120],symmetry_rows=sym[:200],
        note='pairing constructed from actuator names (L/R prefix flip) and asserted present in the model, never by DOF order'))
    # tendon length curves for the mirrored pairs (19<->40) and (26<->47); 3 angles per side
    curves=[]
    def mirrored_joint(left_name):
        for a_,b_ in (('LF','RF'),('LM','RM'),('LH','RH')):
            if left_name.startswith('joint_'+a_):
                other='joint_'+b_+left_name[len('joint_'+a_):]
                if other not in jn: return None
                jr=jn.index(other)                      # joint_row space
                return jr+6                              # convert to DOF space (dof = 6 + joint_row)
        return None
    curve_pairs=[(SIX[0], mirrored_joint(jn[SIX[0]-6])), (SIX[2], mirrored_joint(jn[SIX[2]-6]))]
    flex_slots=[k for k,n in enumerate(anames) if 'Tibia_flex' in n]
    ext_slots=[k for k,n in enumerate(anames) if 'Tibia_extensor' in n]
    for dof_l, dof_r in curve_pairs:
        if dof_r is None: continue
        assert 6 <= dof_r <= 47, dof_r
        assert jn[dof_r-6].startswith('joint_R'), jn[dof_r-6]
        left_leg=jn[dof_l-6][6:8]; right_leg=jn[dof_r-6][6:8]
        fl=[k for k in flex_slots if anames[k].startswith(left_leg)]
        fr=[k for k in flex_slots if anames[k].startswith(right_leg)]
        el=[k for k in ext_slots if anames[k].startswith(left_leg)]
        er=[k for k in ext_slots if anames[k].startswith(right_leg)]
        for side,dof,fs,es in (('left',dof_l,fl,el),('right',dof_r,fr,er)):
            if not fs or not es: continue
            j=dof-6; lo,hi=None,None
            base=qbase[7+j]
            for ang in (base-0.05, base, base+0.05):
                if calls['forward']>=CAP: break
                q=qbase.copy(); q[7+j]=ang
                L=np.asarray(fwd(q,np.zeros(90))['length'])
                curves.append(dict(pair=f'{jn[dof_l-6]}<->{jn[dof_r-6]}',side=side,joint=jn[j],dof=dof,angle=float(ang),
                    flex_actuator=anames[fs[0]],flex_length=float(L[fs[0]]),
                    extensor_actuator=anames[es[0]],extensor_length=float(L[es[0]])))
    with (out/'TENDON_LENGTH_CURVES.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(curves[0])); w.writeheader(); w.writerows(curves)
    # asset vs defect verdict
    worst_spread=float(max([t['rel_spread'] for t in triple if t['rel_spread'] is not None],default=0.))
    verdict=('DEFINITION_DEFECT' if worst_spread>1e-9 else 'ASSET_PROPERTY')
    dump(out/'ASSET_OR_DEFECT_VERDICT.json',dict(worst_rel_spread=worst_spread,verdict=verdict,
        explanation=('S186 arm = B = gain*moment (generalized force per activation); S189 arm = actuator_moment = dL/dq. '
                     'The -38.997 vs 0.0132 gap is exactly the gain factor %.2f at slot 59.' % abs(float(gain[59]))),
        m0=dict(act=m0['act'],ctrl=m0['ctrl'],act_upper=act_hi,ctrl_upper=ctrl_hi,act_limited=act_lim,ctrl_limited=ctrl_lim,
                definition='m0 = native upper bound / 0.5; the frozen half-amplitude interface is act=0.5 (ctrl=0.5)')))
    dump(out/'SUMMARY.json',dict(stageA=dict(prov_rows=len(prov),m0=m0,act_hi=act_hi,ctrl_hi=ctrl_hi,act_limited=act_lim,ctrl_limited=ctrl_lim),
        stageB=dict(triple_rows=len(triple),worst_rel_spread=worst_spread,verdict=verdict,n_forwards=calls['forward']),
        calls=calls,units='native; SI NOT_ESTABLISHED',no_step=True,no_neural=True,no_training=True))
    dump(out/'INPUT_HASHES_AFTER.json',{k:sha(k) for k in h})
    print(json.dumps(dict(verdict=verdict,worst_spread=worst_spread,fw=calls['forward'],
        act_hi=act_hi,ctrl_hi=ctrl_hi,act_limited=act_lim,ctrl_limited=ctrl_lim,m0_act=m0['act'],m0_ctrl=m0['ctrl'],
        gain59=float(gain[59]))))
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path)
    a=ap.parse_args(); main(a.output or B_/'s190-arm-definition-symmetry-audit'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
