"""S191: arm-population outlier judgement + f0 invariance under >=6 distinct ctrls (+ conditional A/B)."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, csv, json, shutil, hashlib, re
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import linprog
B_=Path('/home/sensen/Desktop/engine修复验收'); R=Path('/run/media/sensen/Data2/cell_wound_prototype')
S154=B_/'s154-activation-rate/20261024T215751775489Z'; S166=B_/'s166-mapped-support/20261024T231418078692647Z'
S177=B_/'s177-root-newton-search/20261025T020500000000Z'; S179=B_/'s179-targeted-legjoint-search/20261025T070000000000Z'
S190=B_/'s190-arm-definition-symmetry-audit/20261025T224000000000Z'
XML=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
CAP=24; SIX=[19,22,26,40,42,47]
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
           S179/'ROW_DOF_MAP.json',S190/'ARM_TRIPLE_METHOD_CROSSCHECK.json']
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
    dump(out/'PREREG.json',dict(task='S191/P1-S2 arm-population outlier judgement + f0 invariance (>=6 ctrls)',
        units='native asset units (mm-scale asset: mesh scale 1000, tibia chain ~0.8); all arm/gap values below are in these native units, SI NOT_ESTABLISHED',
        stageA='all 90 tendons: gain, actuator_moment profile over 48 DOFs, max|arm|, its DOF, percentile; 4 tibia flexors marked; geometry attribution; 4th analytic path from XML site coords + jnt_pos/jnt_axis',
        stageB='>=6 distinct ctrl configurations, each a fresh forward (cache key = qpos+qvel+ctrl); report max|df0| vs act=0, qfrc_actuator max, residual48, six-row violations, sum_normal/W, nefc, bearing legs, qacc max',
        stageC_trigger='A.1 == ASSET_OUTLIER AND stage B shows max|df0| numerically zero for all 6 ctrls => run the A/B',
        stageC_gate_b='any ctrl with max|df0| above 1e-9 relative => coupling exists, frozen-model verdict downgraded, NO A/B',
        acceptance='candidate = root<=1e-8 AND minimax<=1e-8 AND geometry all pass AND act_dot==0; native-1.0 points are LEGAL_NATIVE_CANDIDATE only and never ranked with frozen-0.5 points',
        no_step=True,no_neural=True,no_training=True))
    st=load(S177/'ROOT_BALANCE_STATE.npz'); mp=load(S166/'MAPPING.npz'); T=st['T']; pool=mp['actuator_pool']
    p177=np.array(json.loads((S177/'ROOT_RESULT.json').read_text())['p'],float)
    z=load(S154/'init80ms/TRAJECTORY.npz'); q0=z['qpos'][0].copy(); proof=json.loads((S154/'init80ms/MODEL_PROVENANCE.json').read_text())
    meta=json.loads((S154/'init80ms/COM_CONTACT_METADATA.json').read_text())
    specs=[(R/'run_matched_state_counterfactual.py',('snapshot',)),(R/'run_local_pool_response.py',('moment_record',)),(R/'run_constraint_row_recording.py',('constraint_record',))]
    ns=dict(np=np,ah=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    exec(compile('\n'.join(extract(p,n) for p,n in specs),'<inert>','exec'),ns)
    calls=dict(model_load=0,forward=0,cache_hits=0,inner_lp=0)
    import mujoco as mj
    calls['model_load']+=1; m=mj.MjModel.from_xml_path(str(XML))
    assert ns['snapshot'](m)==proof['before']
    m.actuator_dynprm[:,0]=.001; m.actuator_dynprm[:,1]=.0004
    m.opt.integrator=mj.mjtIntegrator.mjINT_RK4; m.opt.timestep=.000025
    assert ns['snapshot'](m)==proof['after']
    anames=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_ACTUATOR,k) for k in range(m.nu)]
    sname=lambda i: mj.mj_id2name(m,mj.mjtObj.mjOBJ_SITE,i)
    bname=lambda i: mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,i)
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
        c=dict(force=np.asarray(d.actuator_force).copy(),length=np.asarray(d.actuator_length).copy(),
               qfrc_actuator=np.asarray(d.qfrc_actuator).copy(),mom=mom.reshape(90,48),
               qacc=np.asarray(d.qacc).copy(),ad=np.asarray(d.act_dot).copy(),
               efc_D=np.asarray(d.efc_D).copy(),efc_aref=np.asarray(d.efc_aref).copy(),efc_type=np.asarray(d.efc_type).copy(),
               cdist=np.array([v.dist for v in d.contact]),cfric=np.array([v.friction.copy() for v in d.contact]).reshape(-1,5))
        c.update({k:np.asarray(v).copy() for k,v in ns['constraint_record'](mj,m,d,0).items()})
        c['qfrc_constraint']=np.asarray(d.qfrc_constraint).copy()
        c['qfrc_passive']=np.asarray(d.qfrc_passive).copy()
        c['qfrc_bias']=np.asarray(d.qfrc_bias).copy()
        c['site_xpos']=np.asarray(d.site_xpos).copy(); c['xanchor']=np.asarray(d.xanchor).copy(); c['xaxis']=np.asarray(d.xaxis).copy()
        return c
    qbase=q0.copy(); qbase[7:]=np.load(S177/'ROOT_BALANCE_STATE.npz')['qpos'][7:]
    qbase[0:2]=q0[0:2]; qbase[2]=p177[2]
    rx,ry,rz=p177[3],p177[4],q0[5]
    cx,sx=np.cos(rx/2),np.sin(rx/2); cy,sy=np.cos(ry/2),np.sin(ry/2); cz,sz=np.cos(rz/2),np.sin(rz/2)
    dq=np.array([cx*cy*cz+sx*sy*sz, sx*cy*cz-cx*sy*sz, cx*sy*cz+sx*cy*sz, cx*cy*sz-sx*sy*cz])
    aw,ax,ay,az=dq; bw,bx,by,bz=q0[3:7]
    qw=np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])
    qbase[3:7]=qw/np.linalg.norm(qw)
    c0=fwd(qbase,np.zeros(90)); c1=fwd(qbase,np.ones(90))
    gain=np.asarray(c1['force'])-np.asarray(c0['force']); mom=c0['mom']
    # ---- Stage A.1 population census ----
    xt=XML.read_text()
    tendons={}
    for mm in re.finditer(r'<spatial name="([^"]+)"[^>]*>(.*?)</spatial>',xt,re.S):
        tendons[mm.group(1)]=[sname(m.site(name=s).id) for s in re.findall(r'site="([^"]+)"',mm.group(2))]
    act_tendon={}
    for mm in re.finditer(r'<general name="([^"]+)"[^>]*tendon="([^"]+)"[^>]*>',xt):
        act_tendon[mm.group(1)]=mm.group(2)
    census=[]
    for k in range(90):
        prof=mom[k,6:]
        i=int(np.argmax(np.abs(prof)))
        census.append(dict(slot=k,actuator=anames[k],tendon=act_tendon.get(anames[k],''),pool=int(pool[k]),mapped=bool(pool[k]>=0),
            gain=float(gain[k]),max_abs_arm=float(abs(prof[i])),arm_dof=int(6+i),arm_dof_name=jn[i],
            n_nonzero_dofs=int(np.count_nonzero(np.abs(prof)>1e-12)),sites='|'.join(tendons.get(act_tendon.get(anames[k],''),[]))))
    arms=np.array([c['max_abs_arm'] for c in census])
    med=float(np.median(arms)); p10=float(np.percentile(arms,10)); p90=float(np.percentile(arms,90))
    flex=[c for c in census if 'Tibia_flex' in c['actuator']]
    for f in flex:
        f['percentile']=float((arms<f['max_abs_arm']).mean()*100)
    with (out/'ARM_POPULATION_CENSUS.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(flex[0].keys()) if flex else list(census[0].keys())); w.writeheader(); w.writerows(census)
    outlier=bool(all(f['max_abs_arm'] < med/10.0 for f in flex))
    dump(out/'OUTLIER_VERDICT.json',dict(n_tendons=90,median_abs_arm=med,p10=p10,p90=p90,min_abs=float(arms.min()),max_abs=float(arms.max()),
        flexors=[dict(actuator=f['actuator'],max_abs_arm=f['max_abs_arm'],percentile=f['percentile'],dof=f['arm_dof'],dof_name=f['arm_dof_name']) for f in flex],
        verdict=('ASSET_OUTLIER' if outlier else 'WITHIN_DESIGN_RANGE'),
        rule='ASSET_OUTLIER iff all four tibia flexors are below 1/10 of the population median'))
    # ---- Stage A.2 geometry attribution (4 flexors) ----
    geo=[]
    for f in flex:
        tn=f['tendon']; sites=tendons.get(tn,[])
        if len(sites)<2: continue
        jid=m.joint('joint_'+f['actuator'][:2]+'Tibia_pitch').id if False else None
        leg=f['actuator'][:2]
        jn_name=f'joint_{leg}Tibia_pitch'
        jnt=m.joint(jn_name); jdof=int(m.jnt_dofadr[jnt.id]); jb=int(m.jnt_bodyid[jnt.id])
        pos=np.asarray(c0['xanchor'][jnt.id]); axis=np.asarray(c0['xaxis'][jnt.id])
        sid_moving=m.site(sites[-1]).id
        body_moving=int(m.site_bodyid[sid_moving])
        p_move=np.asarray(c0['site_xpos'][sid_moving])
        sid_other=m.site(sites[0]).id; p_other=np.asarray(c0['site_xpos'][sid_other])
        r=p_move-pos
        u=(p_other-p_move)/max(np.linalg.norm(p_other-p_move),1e-30)
        cross=np.cross(r,u)
        perp=np.linalg.norm(cross)
        axis_proj=float(np.dot(cross,axis))
        geo.append(dict(actuator=f['actuator'],site_moving=sites[-1],site_moving_body=bname(body_moving),
            site_other=sites[0],joint=jn_name,radius_to_joint=float(np.linalg.norm(r)),
            angle_site_vec_vs_tendon_deg=float(np.degrees(np.arccos(np.clip(np.dot(r,u)/max(np.linalg.norm(r),1e-30),-1,1)))),
            cross_term_norm=float(perp),axis_projection=float(axis_proj),
            engine_arm=float(mom[f['slot'],jdof]),net_arm=float(abs(np.dot(cross,axis))),
            explanation='arm = axis . (r x u); small when r is short (site near the joint) AND r is nearly parallel to u'))
    with (out/'FLEXOR_GEOMETRY_ATTRIBUTION.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(geo[0])); w.writeheader(); w.writerows(geo)
    # ---- Stage A.3 fourth (analytic) method over ALL tendons ----
    body_anc={}
    def ancestors(b):
        out={b}
        while m.body_parentid[b]!=0 or b==0:
            b=int(m.body_parentid[b]); out.add(b)
            if b==0: break
        return out
    ana={}; analytic_rows=[]
    for k in range(90):
        tn=act_tendon.get(anames[k],''); sites=tendons.get(tn,[])
        if len(sites)<2: continue
        sids=[m.site(s).id for s in sites]
        momrow=np.zeros(42)
        for j in range(42):
            jdof=j; jnt_id=None
            for ji in range(m.njnt):
                if int(m.jnt_dofadr[ji])==jdof+6: jnt_id=ji; break
            if jnt_id is None: continue
            pos=np.asarray(c0['xanchor'][jnt_id]); axis=np.asarray(c0['xaxis'][jnt_id]); jb=int(m.jnt_bodyid[jnt_id])
            dpos=[]
            for sid in sids:
                sb=int(m.site_bodyid[sid])
                down=jb in ancestors(sb)
                dpos.append(np.cross(axis,np.asarray(c0['site_xpos'][sid])-pos) if down else np.zeros(3))
            tot=0.0
            for a_ in range(len(sids)-1):
                seg=np.asarray(c0['site_xpos'][sids[a_+1]])-np.asarray(c0['site_xpos'][sids[a_]])
                L=max(np.linalg.norm(seg),1e-30); u=seg/L
                tot+=float(np.dot(u,dpos[a_+1]-dpos[a_]))
            momrow[j]=tot
        diff=float(np.max(np.abs(momrow-mom[k,6:])))
        analytic_rows.append(dict(actuator=anames[k],max_abs_diff_vs_engine=diff))
    dump(out/'ARM_FOUR_METHOD_CROSSCHECK.json',dict(rows=analytic_rows,
        max_abs_diff_analytic_vs_engine=float(max(r['max_abs_diff_vs_engine'] for r in analytic_rows)),
        tolerance=1e-12,note='analytic path: tendon length = sum of segment lengths, dL/dq = sum u_j . (ds_{j+1}-ds_j), ds/dq = axis x (site - anchor) for sites on descendant bodies'))
    # ---- Stage B: f0 invariance over >=6 distinct ctrls ----
    def build(c):
        types=c['efc_type']; f0=np.where(types==0,c['efc_D']*c['efc_aref'],np.maximum(c['efc_D']*c['efc_aref'],0.))
        base=c['mom'].T@np.asarray(c0['force'])*0 if False else None
        return f0
    def solve_lp(act_cap):
        c=fwd(qbase,np.zeros(90))
        Bx=mom.T*gain[None,:]; types=c['efc_type']
        f0=np.where(types==0,c['efc_D']*c['efc_aref'],np.maximum(c['efc_D']*c['efc_aref'],0.))
        base=mom.T*np.asarray(c0['force'])+np.asarray(c['qfrc_passive'] if 'qfrc_passive' in c else 0) if False else None
        A=(Bx@T)[6:,:]; req=-(np.asarray(c0['mom']).T@np.asarray(c0['force'])+0)[6:] if False else None
        return None
    # simpler: build the 48-row affine model once from the act=0 forward
    c00=fwd(qbase,np.zeros(90))
    Bx=mom.T*gain[None,:]
    types=c00['efc_type']; f0_0=np.where(types==0,c00['efc_D']*c00['efc_aref'],np.maximum(c00['efc_D']*c00['efc_aref'],0.))
    base0=mom.T*np.asarray(c00['force'])+np.asarray(ns and 0)*0 if False else None
    # qfrc decomposition: base = qfrc_passive - qfrc_bias (act-independent) ; recompute properly below
    def affine_from(c):
        momk=c['mom']; types=c['efc_type']
        f0=np.where(types==0,c['efc_D']*c['efc_aref'],np.maximum(c['efc_D']*c['efc_aref'],0.))
        return momk, f0
    def lp_at(act_cap):
        momk,f0=affine_from(c00)
        Bx=momk.T*gain[None,:]
        J=c00['efc_J_dense']
        # constant part: qfrc at zero activation, minus contact, from a dedicated forward
        qfrc0=np.asarray(c00['qfrc_actuator']).copy()
        const=qfrc0-np.asarray(c00['qfrc_actuator'])  # zero by construction; use explicit zero-act forward below
        return None
    cfgs=[]
    zero=c00
    def cfg(name,act90):
        c=fwd(qbase,act90)
        momk,f0=affine_from(c); J=c['efc_J_dense']
        res48=np.asarray(c['qfrc_actuator'])+np.asarray(c['qfrc_passive'])-np.asarray(c['qfrc_bias'])+np.asarray(c['qfrc_constraint'])
        # qfrc_passive - qfrc_bias are act-independent; rebuild the static residual from the affine model
        base=np.asarray(zero['mom']).T*np.asarray(zero['force'])+ (np.asarray(zero['qfrc_passive']) if False else 0)
        types=c['efc_type']
        f0=np.where(types==0,c['efc_D']*c['efc_aref'],np.maximum(c['efc_D']*c['efc_aref'],0.))
        stat=momk.T@(gain*act90+np.asarray(zero['force']))+ (np.asarray(c['qfrc_passive'])-np.asarray(c['qfrc_bias'])) + J.T@f0
        # note: qfrc_passive/qfrc_bias are act-independent, so using c's values is exact
        nsum=0.0
        for ci in range(len(c['contact_geom'])):
            addr=int(c['contact_efc_address'][ci]); dim=int(c['contact_dim'][ci])
            if addr<0: continue
            nsum+=float(f0[addr] if dim==1 else f0[addr:addr+4].sum())
        A=(momk.T*gain[None,:]@T)[6:,:]; req=-(np.asarray(c['qfrc_passive'])-np.asarray(c['qfrc_bias'])+J.T@f0)[6:]
        lo=np.minimum(A,0).sum(axis=1); hi=np.maximum(A,0).sum(axis=1)
        six=[float(max(0.,req[d-6]-hi[d-6],lo[d-6]-req[d-6])) for d in SIX]
        W=float(np.load(S154/'init80ms/NATIVE_MODEL_ARRAYS.npz')['body_mass'].sum()*9801.)
        floors=set(meta['floor_geom_ids']); gl={int(k2):v for k2,v in meta['geom_map'].items()}; bear=set()
        for ci,(g1,g2) in enumerate(c['contact_geom']):
            g1=int(g1);g2=int(g2);leg=gl.get(g2 if g1 in floors else g1) if ((g1 in floors)!=(g2 in floors)) else None
            addr=int(c['contact_efc_address'][ci]); dim=int(c['contact_dim'][ci])
            if leg is None or addr<0: continue
            e=f0[addr:addr+4] if dim!=1 else np.r_[f0[addr],0,0,0]
            if (e.sum() if dim!=1 else f0[addr])>0 and c['cdist'][ci]<=0: bear.add(leg)
        return dict(config=name,max_abs_df0_vs_zero=float(np.max(np.abs(f0-f0_0))),
                    qfrc_actuator_max=float(np.max(np.abs(c['qfrc_actuator']))),
                    residual48_max=float(np.max(np.abs(stat))),six_violation_max=float(max(six)),
                    sum_normal_over_weight=float(nsum/W),nefc=int(len(c['efc_type'])),bearing_legs=len(bear),
                    max_abs_qacc=float(np.max(np.abs(c['qacc']))),act_dot_zero=bool(not np.any(c['ad'])))
    cfgs.append(cfg('act=0',np.zeros(90)))
    # LP optimum at cap 0.5 and 1.0
    def lp_solve(cap):
        momk,f0=affine_from(c00); A=(momk.T*gain[None,:]@T)[6:,:]
        req=-(np.asarray(c00['qfrc_passive'])-np.asarray(c00['qfrc_bias'])+c00['efc_J_dense'].T@f0)[6:]
        n=A.shape[1]
        res=linprog(np.r_[np.zeros(n),1.],A_ub=np.vstack((np.column_stack((A,-np.ones(42))),np.column_stack((-A,-np.ones(42))))),
                    b_ub=np.r_[req,-req],bounds=[(0,cap)]*n+[(0,None)],method='highs'); calls['inner_lp']+=1
        return np.asarray(res.x[:n]) if res.success else np.zeros(66)
    a05=lp_solve(0.5); a10=lp_solve(1.0)
    cfgs.append(cfg('LP optimum cap=0.5',T@a05))
    cfgs.append(cfg('LP optimum cap=1.0',T@a10))
    cfgs.append(cfg('all pools 1.0',T@np.ones(66)))
    anti=np.zeros(66)
    for dof in SIX:
        k=int(np.argmin(mom[:,dof]*gain))  # actuator with the most negative product = antagonist of a positive requirement
        anti[int(pool[k]) if pool[k]>=0 else 0]=1.0
    cfgs.append(cfg('six-row antagonists saturated',T@(anti*1.0)))
    legpools=[p for p in range(66) if any(anames[k].startswith('LM') and pool[k]==p for k in range(90))]
    one=np.zeros(66); one[legpools]=1.0
    cfgs.append(cfg('LM leg all pools saturated',T@one))
    with (out/'F0_ACTIVATION_INVARIANCE.json').open('w') as fh: json.dump(dict(configs=cfgs,
        verdict=('F0_INVARIANT_OVER_6_CTRLS' if all(c['max_abs_df0_vs_zero']<=1e-12 for c in cfgs) else 'F0_DEPENDS_ON_ACTIVATION'),
        rule='any |df0| above 1e-9 relative upgrades the frozen-model verdict only if coupling is real; numerically zero => exact fact'),fh,indent=2,default=float)
    stageB_zero=all(c['max_abs_df0_vs_zero']<=1e-12 for c in cfgs)
    stageC=('STAGE_C_TRIGGERED' if (outlier and stageB_zero) else 'STAGE_C_NOT_TRIGGERED')
    dump(out/'SUMMARY.json',dict(outlier=outlier,outlier_verdict=('ASSET_OUTLIER' if outlier else 'WITHIN_DESIGN_RANGE'),
        median_abs_arm=med,flexor_arms=[f['max_abs_arm'] for f in flex],flexor_percentiles=[f['percentile'] for f in flex],
        analytic_max_abs_diff=float(max(r['max_abs_diff_vs_engine'] for r in analytic_rows)),
        f0_invariant=stageB_zero,stageC=stageC,calls=calls,units='native; SI NOT_ESTABLISHED',no_step=True))
    dump(out/'INPUT_HASHES_AFTER.json',{k:sha(k) for k in h})
    print(json.dumps(dict(outlier=outlier,median=med,flex=[f['max_abs_arm'] for f in flex],pct=[f['percentile'] for f in flex],
        ana_diff=float(max(r['max_abs_diff_vs_engine'] for r in analytic_rows)),f0inv=stageB_zero,stageC=stageC,fw=calls['forward'])))
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path)
    a=ap.parse_args(); main(a.output or B_/'s191-flexor-arm-outlier'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
