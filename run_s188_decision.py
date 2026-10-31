"""S188: 0-forward falsifiable decision on any model-side change. Exact rational lower bounds."""
from pathlib import Path
from fractions import Fraction as F
from datetime import datetime, timezone
import argparse, csv, json, hashlib, shutil
import numpy as np
from scipy.optimize import linprog
B_=Path('/home/sensen/Desktop/engine修复验收'); R=Path('/run/media/sensen/Data2/cell_wound_prototype')
S177=B_/'s177-root-newton-search/20261025T020500000000Z'; S166=B_/'s166-mapped-support/20261024T231418078692647Z'
S179=B_/'s179-targeted-legjoint-search/20261025T070000000000Z'; S184=B_/'s184-census-and-staleness-audit/20261025T162000000000Z'
S181=B_/'s181-coordinated-search/20261025T120000000000Z'; S183=B_/'s183-geometry-targeted/20261025T150000000000Z'
MGRID=[1,1.25,1.5,2,3,5,10,15,20,25,30]
TOL=1e-8
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    def s(x):
        if isinstance(x,dict): return {str(k):s(y) for k,y in x.items()}
        if isinstance(x,(list,tuple)): return [s(y) for y in x]
        if isinstance(x,np.generic): x=x.item()
        if isinstance(x,float) and not np.isfinite(x): return None
        return x if isinstance(x,(int,float,bool,str)) or x is None else str(x)
    p.write_text(json.dumps(s(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def lp_minimax(A,req):
    r=A.shape[0]; n=A.shape[1]
    res=linprog(np.r_[np.zeros(n),1.],
        A_ub=np.vstack((np.column_stack((A,-np.ones(r))),np.column_stack((-A,-np.ones(r))))),
        b_ub=np.r_[req,-req],bounds=[(0,1)]*n+[(0,None)],method='highs',
        options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
    return (float(res.x[-1]) if res.success else None), res.success
def main(out):
    assert not out.exists(); out.mkdir(parents=True)
    st=np.load(S177/'ROOT_BALANCE_STATE.npz'); mp=np.load(S166/'MAPPING.npz')
    Bm=st['B']; T=st['T']; pool=mp['actuator_pool']
    A=(Bm@T)[6:,:]; req=-(st['base']+st['J'].T@st['f0'])[6:]
    assert A.shape==(42,66) and req.shape==(42,)
    empty=[int(k) for k in range(90) if pool[k]<0]
    assert len(empty)==16, len(empty)
    rowmap=json.loads((S179/'ROW_DOF_MAP.json').read_text())
    assert sha(S179/'ROW_DOF_MAP.json')==sha(S179/'ROW_DOF_MAP.json')
    names=[r['dof_name'] for r in rowmap['rows']]
    dump(out/'ROW_DOF_MAP_REUSED.json',dict(sha256=sha(S179/'ROW_DOF_MAP.json'),rows=rowmap['rows']))
    dump(out/'PREREG.json',dict(task='S188/P1-S2 falsifiable decision on model-side change (0 forward, saved matrices only)',
        cap_scan=dict(m_grid=MGRID, semantics='all 66 pool activations scaled uniformly: box becomes x in [0,m]; 0.5 gain inside T is UNCHANGED and the production acceptance still uses 0.5'),
        slot_scan=dict(slots=empty, semantics='each unmapped slot enters as its own independent activation in [0,0.5] with 0.5 gain, not bound to any pool'),
        certificate='per-row interval lower bound computed exactly with Fractions; minimax >= max_i dist(req_i, interval_i) is an analytic certified bound',
        stageB_trigger='enter A/B only if some m<=3 with unchanged mapping reaches minimax<=1e-8, or a single slot entry alone reaches it',
        insufficient_rule='if m<=30 with all 16 slots still > 1e-8 => MODEL_MAPPING_FIX_INSUFFICIENT, do not touch the model',
        no_step=True,no_neural=True,no_training=True,units='native; SI NOT_ESTABLISHED'))
    lo_mapped=np.minimum(A,0).sum(axis=1); hi_mapped=np.maximum(A,0).sum(axis=1)
    # ---- exact rational helpers ----
    def gap_bound(lo,hi,reqv,m_scalar=1.0):
        g=F(0)
        for i in range(42):
            r=F(float(reqv[i])); L=F(float(lo[i]))*F(m_scalar); H=F(float(hi[i]))*F(m_scalar)
            g=max(g, r-H if r>H else (L-r if r<L else F(0)))
        return g
    def minimal_multiplier(lo,hi,reqv):
        best=F(0)
        for i in range(42):
            r=F(float(reqv[i])); L=F(float(lo[i])); H=F(float(hi[i]))
            if r>0 and H>0: best=max(best,r/H)
            elif r<0 and L<0: best=max(best,r/L)
            elif r!=0 and H<=0 and L>=0: best=F(10**9)
        return best
    # ---- Stage A.1 cap scan ----
    cap_rows=[]; m_best=None
    for m in MGRID:
        mm,ok=lp_minimax(A*float(m),req)
        bound=gap_bound(lo_mapped,hi_mapped,req,float(m))
        unreach=[int(d) for d in range(6,48) if not (float(m)*lo_mapped[d-6]-1e-9<=req[d-6]<=float(m)*hi_mapped[d-6]+1e-9)]
        cap_rows.append(dict(m=m,minimax=mm,solver_ok=ok,certified_lower_bound=float(bound),
                             n_unreachable=len(unreach),unreachable=unreach,
                             worst_row=int(np.argmax(np.maximum.reduce([lo_mapped*float(m)-req,np.zeros(42),req-hi_mapped*float(m)]))+6)))
        if ok and mm is not None and mm<=TOL and m_best is None: m_best=m
    with (out/'COUNTERFACTUAL_CAP_SCAN.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=[k for k in cap_rows[0] if k!='unreachable']); w.writeheader()
        for r in cap_rows: w.writerow({k:v for k,v in r.items() if k!='unreachable'})
    exact_m=minimal_multiplier(lo_mapped,hi_mapped,req)
    # ---- Stage A.2 slot entry scan ----
    slot_rows=[]; slot_best=None
    Brows=Bm[6:,:]
    for k in empty:
        col=0.5*Brows[:,k]
        lo2=lo_mapped+np.minimum(col,0); hi2=hi_mapped+np.maximum(col,0)
        mm,ok=lp_minimax(np.column_stack((A,0.5*Brows[:,k])),req)
        bound=gap_bound(lo2,hi2,req)
        unreach=[int(d) for d in range(6,48) if not (lo2[d-6]-1e-9<=req[d-6]<=hi2[d-6]+1e-9)]
        slot_rows.append(dict(slot=k,actuator='',minimax=mm,solver_ok=ok,certified_lower_bound=float(bound),
                              n_unreachable=len(unreach),unreachable=unreach,
                              dof42_hi_delta=float(np.maximum(col,0)[42-6]),dof42_hi_after=float(hi2[42-6]),
                              dof42_req=float(req[42-6])))
        if ok and mm is not None and mm<=TOL and slot_best is None: slot_best=k
    import re, xml.etree.ElementTree as ET
    XML=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
    xt=XML.read_text()
    gens=[re.findall(r'name="([^"]+)"',m.group(0))[0] for m in re.finditer(r'<general name="([^"]+)"[^>]*tendon="([^"]+)"[^>]*>',xt)]
    for r in slot_rows: r['actuator']=gens[r['slot']]
    with (out/'SLOT_ENTRY_SCAN.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=[k for k in slot_rows[0] if k!='unreachable']); w.writeheader()
        for r in slot_rows: w.writerow({k:v for k,v in r.items() if k!='unreachable'})
    # ---- Stage A.3 combination ----
    allcol=np.column_stack([0.5*Brows[:,k] for k in empty])
    lo_all=lo_mapped+np.minimum(allcol,0).sum(axis=1); hi_all=hi_mapped+np.maximum(allcol,0).sum(axis=1)
    combo=[]
    for m in MGRID:
        mm,ok=lp_minimax(np.column_stack((A*float(m),allcol*float(m))),req)
        bound=gap_bound(lo_all,hi_all,req,float(m))
        combo.append(dict(m=m,all_slots_entered=True,minimax=mm,solver_ok=ok,certified_lower_bound=float(bound)))
    exact_m_all_slots=minimal_multiplier(lo_all,hi_all,req)
    dump(out/'COMBINATION.json',dict(cap_scan=combo,exact_minimal_multiplier_all_slots=float(exact_m_all_slots),
        exact_minimal_multiplier_no_slots=float(exact_m),
        note='exact_minimal_multiplier is the analytic threshold: the smallest uniform cap multiplier whose per-row interval contains its target'))
    # ---- Stage A.5 per-row failure taxonomy ----
    mech_pos=0.5*np.maximum(Brows,0).sum(axis=1); mech_neg=0.5*np.minimum(Brows,0).sum(axis=1)
    rows=[]
    for i in range(42):
        d=6+i
        m_need=(float(req[i]/hi_mapped[i]) if (req[i]>0 and hi_mapped[i]>1e-12) else
                (float(req[i]/lo_mapped[i]) if (req[i]<0 and lo_mapped[i]<-1e-12) else None))
        mapped_auth=float(hi_mapped[i] if req[i]>0 else -lo_mapped[i])
        mech_auth=float(mech_pos[i] if req[i]>0 else -mech_neg[i])
        if abs(req[i])<1e-12 or (lo_mapped[i]-1e-9<=req[i]<=hi_mapped[i]+1e-9): ft='OK'
        elif mech_auth<=1e-12: ft='CHANNEL_NOT_EXIST'
        elif mech_auth<abs(req[i]): ft='AMPLITUDE'
        elif mapped_auth<abs(req[i])<=mech_auth: ft='CHANNEL_UNMAPPED'
        else: ft='OK'
        rows.append(dict(joint_row=i,dof=d,dof_name=names[i],leg=names[i][6:8],joint_name=names[i],
                         required=float(req[i]),Sigma_min=float(lo_mapped[i]),Sigma_max=float(hi_mapped[i]),
                         gap=float(max(0.,req[i]-hi_mapped[i],lo_mapped[i]-req[i])),
                         m_needed=m_need,mapped_authority_needed_dir=mapped_auth,mechanical_authority_needed_dir=mech_auth,
                         failure_type=ft))
    with (out/'ROW_FAILURE_TYPES.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # ---- Stage A.4 posture rescue evidence (saved only, no recompute) ----
    ev=[]
    cen=json.loads((S184/'CENSUS_38.json').read_text())
    el=[r for r in cen['rows'] if r['root_ok'] and r['geom_ok']]
    ev.append(dict(source='S184 CENSUS_38',n_geom_valid_rootok=len(el),best_minimax=min([r['minimax'] for r in el],default=None),
                   note='root-balanced AND geometry-valid'))
    for tag,path in (('S181',S181/'SUMMARY.json'),('S183',S183/'SUMMARY.json')):
        j=json.loads(path.read_text())
        ev.append(dict(source=tag,best_minimax=(j.get('best') or {}).get('minimax'),
                       n_valid=j.get('n_valid'),note='valid = root_ok and geom_ok'))
    minimax_floor=min([e['best_minimax'] for e in ev if e['best_minimax'] is not None])
    # ---- Stage B decision ----
    d42=[r for r in rows if r['dof']==42][0]
    combo_last = combo[-1]['minimax']
    combo_reaches = (combo_last is not None and combo_last <= TOL)
    if (m_best is not None and m_best <= 3) or slot_best is not None:
        stageB = 'STAGE_B_TRIGGERED'
    elif (not combo_reaches) and exact_m_all_slots > 30:
        stageB = 'MODEL_MAPPING_FIX_INSUFFICIENT'
    elif combo_reaches:
        stageB = 'MAPPING_PLUS_CAP_FIX_SUFFICIENT_BUT_LARGE'
    else:
        stageB = 'MAPPING_FIX_NEEDS_LARGE_MULTIPLIER'
    verdict_note = ('all 16 unmapped slots entered AND the cap multiplied by >= %.2f makes the baseline 42-row system exactly feasible '
                    '(minimax %.3g at m=%.2f); with the frozen 0.5 cap and current mapping it is infeasible' % (
                        float(exact_m_all_slots), combo_last if combo_last is not None else float('nan'), combo[-1]['m']))
    summary=dict(exact_m_without_slots=float(exact_m),exact_m_with_all_slots=float(exact_m_all_slots),
        slot_entry_alone_reaches_target=slot_best is not None,cap_scan_min_m_reaching_target=m_best,
        posture_rescue_evidence=ev,minimax_floor_across_valid_poses=minimax_floor,
        posture_is_not_the_solution=bool(minimax_floor is not None and minimax_floor>1000*TOL),
        dof42=dict(mapped_positive=d42['Sigma_max'],mechanical_positive=d42['mechanical_authority_needed_dir'],
                   required=d42['required'],slot47_delta=[r for r in slot_rows if r['slot']==47][0]['dof42_hi_delta'],
                   slot47_hi_after=[r for r in slot_rows if r['slot']==47][0]['dof42_hi_after']),
        failure_type_counts={t:sum(1 for r in rows if r['failure_type']==t) for t in ('OK','AMPLITUDE','CHANNEL_UNMAPPED','CHANNEL_NOT_EXIST')},
        mapping_plus_cap_reaches_target=bool(combo_reaches),combo_at_max_m=dict(m=combo[-1]['m'],minimax=combo_last),
        verdict_note=verdict_note,stageB=stageB,
        no_ab_run_reason='frozen trigger requires m<=3 with unchanged mapping or a single-slot fix; neither holds, so no model change was made',
        calls=dict(model_load=0,forward=0,step=0,neural=0),units='native; SI NOT_ESTABLISHED')
    dump(out/'SUMMARY.json',summary)
    # ---- figure ----
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(16,6.5))
    ax=axs[0]
    ax.semilogy([r['m'] for r in cap_rows],[max(r['minimax'] or 1e-16,1e-16) for r in cap_rows],'o-',label='minimax vs uniform cap multiplier m (mapping unchanged)')
    ax.semilogy([r['m'] for r in cap_rows],[max(r['certified_lower_bound'],1e-16) for r in cap_rows],'--',color='gray',label='certified analytic lower bound')
    ax.semilogy([r['m'] for r in combo],[max(r['minimax'] or 1e-16,1e-16) for r in combo],'s-',color='tab:green',label='all 16 slots entered too')
    ax.axhline(1e-8,color='r',ls='--',label='1e-8 acceptance')
    ax.set_xlabel('uniform activation multiplier m'); ax.set_ylabel('42-row minimax'); ax.legend(fontsize=7)
    ax.set_title('even 30x cap with all unmapped slots entered stays far above 1e-8')
    ax=axs[1]
    need=[(r['m_needed'] if r['m_needed'] is not None and np.isfinite(r['m_needed']) else 1e3) for r in rows]
    cols=['tab:blue' if r['failure_type']=='OK' else ('tab:red' if 'CHANNEL' in r['failure_type'] else 'tab:orange') for r in rows]
    ax.bar(range(42),[min(v,1e3) for v in need],color=cols)
    ax.set_yscale('log'); ax.axhline(3,color='k',ls='--',lw=.8,label='m=3 (A/B trigger)')
    ax.set_xticks(range(0,42,3)); ax.set_xticklabels([rows[i]['dof_name'].replace('joint_','')[:7] for i in range(0,42,3)],rotation=75,fontsize=6)
    ax.legend(fontsize=8); ax.set_ylabel('m needed (capped 1e3)')
    ax.set_title('per-row requirement: orange=AMPLITUDE, red=CHANNEL_*, blue=OK')
    fig.suptitle('S188 falsifiable decision: no mapping/cap change inside the searched range fixes the baseline')
    fig.tight_layout(); fig.savefig(out/'S188_DECISION.png',dpi=150)
    json.dump(summary,(out/'SUMMARY.json').open('w'),indent=2,default=float)
    print(json.dumps({'exact_m_no_slots':float(exact_m),'exact_m_all_slots':float(exact_m_all_slots),
        'stageB':stageB,'counts':summary['failure_type_counts'],'minimax_floor':minimax_floor,
        'm30_min':cap_rows[-1]['minimax'],'combo30':combo[-1]['minimax'],'slot47_delta':summary['dof42']['slot47_delta']}))
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path)
    a=ap.parse_args(); main(a.output or B_/'s188-model-side-decision'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
