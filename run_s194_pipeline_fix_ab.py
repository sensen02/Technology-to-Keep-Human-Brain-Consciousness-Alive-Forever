"""S194: pipeline-fix prototype (no isotropic site scaling for the tibia flexors) + frozen A/B.

Variant B: the four mid/hind tibia-flexor site offsets are set to the SOURCE FlyGym LF offsets
(relative to the joint at the body origin), un-scaled and mirrored by the body name.
Only 4x3 site pos values change; the original XML is never modified.
"""
from pathlib import Path
import argparse, ast, json, shutil, hashlib, re, csv
import numpy as np
from scipy.optimize import linprog
from root_newton_core import RootNewton
B_=Path('/home/sensen/Desktop/engine修复验收'); R=Path('/run/media/sensen/Data2/cell_wound_prototype')
S154=B_/'s154-activation-rate/20261024T215751775489Z'; S166=B_/'s166-mapped-support/20261024T231418078692647Z'
S177=B_/'s177-root-newton-search/20261025T020500000000Z'; S179=B_/'s179-targeted-legjoint-search/20261025T070000000000Z'
XML=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
CAP=400; SIX=[19,22,26,40,42,47]
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
SRC=Path('/run/media/sensen/Data2/cell_wound_prototype/venv_body/lib/python3.12/site-packages/flygym/assets/model/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml')
def main(out):
    assert not out.exists(); out.mkdir(parents=True)
    xt=XML.read_text(); orig=sha(XML); sx=SRC.read_text()
    srcsites={}
    for nm in ('bifemlh_r-P1','bifemlh_r-P2','pp21995'):
        mm=re.search(r'<site name="LFTibia_flex_93434_'+re.escape(nm)+r'" pos="([^"]+)"',sx)
        srcsites[nm]=np.array([float(v) for v in mm.group(1).split()])
    field=[]
    xt2=xt
    for leg in ('LH','LM','RH','RM'):
        sign=+1.0 if leg.startswith('L') else -1.0
        for nm in ('bifemlh_r-P1','bifemlh_r-P2','pp21995'):
            sn=f'{leg}Tibia_flex_93434_{nm}'
            mm=re.search(r'<site name="'+re.escape(sn)+r'" pos="([^"]+)"',xt2)
            old=np.array([float(v) for v in mm.group(1).split()])
            new=srcsites[nm].copy(); new[1]*=sign
            xt2=xt2.replace(f'<site name="{sn}" pos="{mm.group(1)}"',
                            '<site name="%s" pos="%.9f %.9f %.9f"'%(sn,new[0],new[1],new[2]))
            field.append(dict(site=sn,old=old.tolist(),new=new.tolist(),delta=float(np.linalg.norm(new-old))))
    xb=out/'MODEL_B_SOURCE_OFFSETS.xml'; xb.write_text(xt2)
    dump(out/'FIELD_DIFF.json',dict(fields=field,original_sha256=orig,b_sha256=sha(xb),source_xml=str(SRC),
        statement='only the 4x3 mid/hind tibia-flexor site offsets change: they are set to the FlyGym LF source offsets (mirrored by body side); no gain/force/gear/joint/mass edits; ORIGINAL XML UNTOUCHED'))
    z=load(S154/'init80ms/TRAJECTORY.npz'); q0=z['qpos'][0].copy()
    proof=json.loads((S154/'init80ms/MODEL_PROVENANCE.json').read_text())
    mp=load(S166/'MAPPING.npz'); T=mp['T']; pool=mp['actuator_pool']
    p177=np.array(json.loads((S177/'ROOT_RESULT.json').read_text())['p'],float)
    rowmap=json.loads((S179/'ROW_DOF_MAP.json').read_text()); jn=[r['dof_name'] for r in rowmap['rows']]
    specs=[(R/'run_matched_state_counterfactual.py',('snapshot',)),(R/'run_local_pool_response.py',('moment_record',)),(R/'run_constraint_row_recording.py',('constraint_record',))]
    ns=dict(np=np,ah=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    exec(compile('\n'.join(extract(p,n) for p,n in specs),'<inert>','exec'),ns)
    import mujoco as mj
    calls=dict(model_load=0,forward=0,inner_lp=0)
    def side(path,tag):
        m=mj.MjModel.from_xml_path(str(path)); calls['model_load']+=1
        m.actuator_dynprm[:,0]=.001; m.actuator_dynprm[:,1]=.0004
        m.opt.integrator=mj.mjtIntegrator.mjINT_RK4; m.opt.timestep=.000025
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
            mom=np.asarray(ns['moment_record'](m,d)['actuator_moment']); mom=mom[0] if mom.ndim==3 else mom
            c=dict(force=np.asarray(d.actuator_force).copy(),mom=mom.reshape(90,48),qacc=np.asarray(d.qacc).copy(),
                   ad=np.asarray(d.act_dot).copy(),efc_D=np.asarray(d.efc_D).copy(),efc_aref=np.asarray(d.efc_aref).copy(),
                   efc_type=np.asarray(d.efc_type).copy(),cdist=np.array([v.dist for v in d.contact]))
            c.update({k:np.asarray(v).copy() for k,v in ns['constraint_row_recording'.replace('constraint_row_recording','constraint_record')](mj,m,d,0).items()})
            c['qfrc_passive']=np.asarray(d.qfrc_passive).copy(); c['qfrc_bias']=np.asarray(d.qfrc_bias).copy()
            return c
        qbase=q0.copy(); qbase[7:]=load(S177/'ROOT_BALANCE_STATE.npz')['qpos'][7:]
        qbase[0:2]=q0[0:2]; qbase[2]=p177[2]
        rx,ry,rz=p177[3],p177[4],q0[5]
        cx,sx2=np.cos(rx/2),np.sin(rx/2); cy,sy=np.cos(ry/2),np.sin(ry/2); cz,sz=np.cos(rz/2),np.sin(rz/2)
        dqq=np.array([cx*cy*cz+sx2*sy*sz, sx2*cy*cz-cx*sy*sz, cx*sy*cz+sx2*cy*sz, cx*cy*sz-sx2*sy*cz])
        aw,ax,ay,az=dqq; bw,bx,by,bz=q0[3:7]
        qw=np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])
        qbase[3:7]=qw/np.linalg.norm(qw)
        c0=fwd(qbase,np.zeros(90)); c1=fwd(qbase,np.ones(90))
        gain=np.asarray(c1['force'])-np.asarray(c0['force'])
        rows=[]
        # flexor arms per leg for the report
        arms={}
        for leg in ('LF','RF','LH','RH','LM','RM'):
            try:
                jd=int(m.jnt_dofadr[m.joint(f'joint_{leg}Tibia_pitch').id])-6
                k=m.actuator(f'{leg}Tibia_flex_93434').id
                arms[leg]=float(c0['mom'][k,jd+6])
            except Exception: pass
        for cap in (0.5,1.0):
            types=c0['efc_type']; f0=np.where(types==0,c0['efc_D']*c0['efc_aref'],np.maximum(c0['efc_D']*c0['efc_aref'],0.))
            A=(c0['mom'].T*gain[None,:]@T)[6:,:]
            req=-(np.asarray(c0['qfrc_passive'])-np.asarray(c0['qfrc_bias'])+c0['efc_J_dense'].T@f0)[6:]
            n=A.shape[1]
            lp=linprog(np.r_[np.zeros(n),1.],A_ub=np.vstack((np.column_stack((A,-np.ones(42))),np.column_stack((-A,-np.ones(42))))),
                       b_ub=np.r_[req,-req],bounds=[(0,cap)]*n+[(0,None)],method='highs'); calls['inner_lp']+=1
            lo=np.minimum(A,0).sum(axis=1); hi=np.maximum(A,0).sum(axis=1)
            reach=(lo-1e-9<=req)&(req<=hi+1e-9)
            rows.append(dict(side=tag,cap=cap,minimax=float(lp.x[-1]) if lp.success else None,
                unreachable=[int(d) for d in range(6,48) if not reach[d-6]],flexor_arms=arms,
                label=('FROZEN_0.5' if cap==0.5 else 'LEGAL_NATIVE_1.0')))
        return rows
    A_=side(XML,'A_ORIGINAL'); B_=side(xb,'B_SOURCE_OFFSETS')
    dump(out/'AB_RESULTS.json',dict(A=A_,B=B_,calls=calls,
        statement='B is a pipeline-fix prototype; its load-bearing results must not be back-filled onto frozen A; the A/B quantifies whether un-scaling the site offsets closes the arm gap'))
    dump(out/'MODEL_AB_PREREG.json',dict(variant='B sets the 4 mid/hind tibia-flexor site offsets to the FlyGym LF source offsets (mirrored), removing the isotropic segment-length scaling',
        sites=[f['site'] for f in field],rule='candidate = root<=1e-8 AND minimax<=1e-8 AND geometry pass AND act_dot==0; native-1.0 points only LEGAL_NATIVE_CANDIDATE'))
    print(json.dumps(dict(A=[(r['cap'],round(r['minimax'],4) if r['minimax'] else None,len(r['unreachable'])) for r in A_],
        B=[(r['cap'],round(r['minimax'],4) if r['minimax'] else None,len(r['unreachable'])) for r in B_],
        armsA={k:round(v,6) for k,v in A_[0]['flexor_arms'].items()},
        armsB={k:round(v,6) for k,v in B_[0]['flexor_arms'].items()},fw=calls['forward'])))
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); main(ap.parse_args().output)
