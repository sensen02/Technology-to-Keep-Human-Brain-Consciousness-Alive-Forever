"""S147/P1: strictly offline S142/S146 physical 0..250us force/work budget.
No MuJoCo import, model load, forward, step, neural replay or tuning.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify_muscle_reconstruction import evaluate

BASE = Path('/home/sensen/Desktop/engine修复验收')
PROJECT = Path('/run/media/sensen/Data2/cell_wound_prototype')
SOURCES = {
    'S142': BASE/'s142-free-physical-prewarm/20261024T201419378677Z/init80ms',
    'S146': BASE/'s146-secant-pose-free-physical/20261024T203645678947Z/init80ms',
}
XML = Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
FILES = ('TRAJECTORY.npz', 'NATIVE_MODEL_ARRAYS.npz', 'CONSTRAINT_ROWS.npz',
         'MOMENT_METADATA.json', 'CONSTRAINT_ROW_METADATA.json', 'COM_CONTACT_METADATA.json')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p, obj):
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')

def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

def table(p, records):
    with p.open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)

def cumulative(y, t):
    return np.r_[0., np.cumsum((y[:-1]+y[1:])*.5*np.diff(t))]

def check(x, y, name, errors):
    error = float(np.max(np.abs(np.asarray(x)-y)))
    errors[name] = error
    assert error <= 1e-10, (name, error)

def analyze(label, p, nodes):
    a, m, r = [load(p/(name+'.npz')) for name in ('TRAJECTORY', 'NATIVE_MODEL_ARRAYS', 'CONSTRAINT_ROWS')]
    cm = json.loads((p/'CONSTRAINT_ROW_METADATA.json').read_text())
    mm = json.loads((p/'MOMENT_METADATA.json').read_text())
    meta = json.loads((p/'COM_CONTACT_METADATA.json').read_text())
    assert mm['format'] == 'row-addressed sparse CSR' and mm['dense_shape'] == [90,48]
    errors = {}
    t = a['time'][:11]
    check(t, np.arange(11)*25e-6, 'actual_physical_clock', errors)
    check(r['time'][:11], t, 'constraint_clock', errors)
    assert np.array_equal(r['physical_step'][:11], np.arange(11))
    for key in ('gainprm','biasprm','dynprm'):
        vals = [list(map(float,n.get(key).split())) for n in nodes]
        assert np.array_equal(np.array([v+[0.]*(10-len(v)) for v in vals]), m['actuator_'+key])
    assert np.array_equal(np.array([list(map(float,n.get('lengthrange').split())) for n in nodes]),m['actuator_lengthrange'])
    assert np.array_equal(m['actuator_actadr'], np.arange(90))
    active = np.zeros((11,90)); passive = active.copy(); total = active.copy()
    moment = np.zeros((11,90,48))
    contact = np.zeros((11,48)); noncontact = contact.copy()
    counts = []
    for i in range(11):
        for j,node in enumerate(nodes):
            f,g,b = evaluate(a['actuator_length'][i,j],a['actuator_velocity'][i,j],a['act'][i,j],node)
            total[i,j],active[i,j],passive[i,j] = f,g*a['act'][i,j],b
            adr,nnz = int(a['moment_rowadr'][i,j]),int(a['moment_rownnz'][i,j])
            cols = a['moment_colind'][i,adr:adr+nnz]
            assert len(set(cols.tolist())) == nnz and np.all((cols>=0)&(cols<48))
            np.add.at(moment[i,j],cols,a['actuator_moment_raw'][i,adr:adr+nnz])
        pre = f'sample{i:03d}_'
        typ,ids,force = [r[pre+k] for k in ('efc_type','efc_id','efc_force')]
        J = r[pre+'efc_J_dense']
        assert not r[pre+'is_sparse'] and J.shape == (len(typ),48)
        check(J,r[pre+'efc_J_raw'].reshape(-1,48),f'raw_constraint_J_{i}',errors)
        contact_types = [cm['constraint_enum'][k] for k in ('mjCNSTR_CONTACT_FRICTIONLESS','mjCNSTR_CONTACT_PYRAMIDAL','mjCNSTR_CONTACT_ELLIPTIC')]
        mask = np.isin(typ,contact_types)
        assert set(typ.tolist()).issubset(set(cm['constraint_enum'].values()))
        assert np.array_equal(r[pre+'row_contact_index'][mask],ids[mask])
        assert np.all(r[pre+'row_contact_index'][~mask] == -1)
        for row in np.flatnonzero(mask):
            c = int(ids[row]); start = int(r[pre+'contact_efc_address'][c]); dim = int(r[pre+'contact_dim'][c])
            n = 1 if typ[row] == 5 else (2*(dim-1) if typ[row] == 6 else dim)
            assert start <= row < start+n and r[pre+'contact_exclude'][c] == 0
        contact[i] = J[mask].T@force[mask]
        noncontact[i] = J[~mask].T@force[~mask]
        counts.append((int(mask.sum()),int((~mask).sum())))
    check(total,a['actuator_force'][:11],'scalar_total_force',errors)
    check(active+passive,total,'scalar_split',errors)
    check(moment,a['actuator_moment'][:11],'CSR_moment_vs_dense',errors)
    qa = np.einsum('tud,tu->td',moment,active)
    qp = np.einsum('tud,tu->td',moment,passive)
    check(qa+qp,a['qfrc_actuator'][:11],'actuator_generalized_projection',errors)
    check(contact+noncontact,a['qfrc_constraint'][:11],'actual_efc_constraint_projection',errors)
    terms = dict(muscle_active=qa,muscle_passive_bias=qp,joint_etc_passive=a['qfrc_passive'][:11],
                 contact_constraint=contact,noncontact_constraint=noncontact,
                 applied=a['qfrc_applied'][:11],negative_bias=-a['qfrc_bias'][:11])
    rhs = sum(terms.values()); ma = a['Ma'][:11]; residual = ma-rhs
    check(ma,rhs,'all48_Ma_closure',errors)
    mass = m['body_mass']; M = float(mass.sum())
    assert mass[0] == 0 and mass[1] == 0
    jac = np.einsum('b,tbkd->tkd',mass,a['body_com_jacp'][:11])/M
    jd = np.einsum('b,tbkd->tkd',mass,a['body_com_jacdot'][:11])/M
    comaz = (np.einsum('tkd,td->tk',jac,a['qacc'][:11])+np.einsum('tkd,td->tk',jd,a['qvel'][:11]))
    comv = np.einsum('tkd,td->tk',jac,a['qvel'][:11])
    check(comaz,a['com_acceleration_world'][:11],'COM_Jac_acceleration',errors)
    check(comv,a['com_velocity_world'][:11],'COM_Jac_velocity',errors)
    check(a['xfrc_applied'][:11],0,'zero_xfrc',errors)
    rawworld = np.einsum('tcij,tci->tcj',a['contact_frame'][:11],a['contact_force_raw6'][:11,:,:3])
    external = (rawworld*(a['contact_external_sign'][:11]*a['contact_external_mask'][:11]*a['contact_valid_mask'][:11])[:,:,None]).sum(1)
    check(external,a['contact_force_world'][:11],'raw_world_contact_sum',errors)
    gravity = M*np.asarray(meta['gravity_world'])
    check(a['gravity_force_world'][:11],gravity,'gravity_mass',errors)
    applied = a['applied_force_world'][:11]
    check(M*comaz,external+gravity+applied,'COM_external_force_balance',errors)
    power = {k:np.einsum('td,td->t',v,a['qvel'][:11]) for k,v in terms.items()}
    check(power['muscle_active']+power['muscle_passive_bias'],(total*a['actuator_velocity'][:11]).sum(1),'muscle_power_virtual_work',errors)
    works = {k:cumulative(v,t) for k,v in power.items()}
    impulses = {k:cumulative(v,t) for k,v in dict(contact=external[:,2],gravity=np.full(11,gravity[2]),applied=applied[:,2]).items()}
    dp = M*(comv[:,2]-comv[0,2]); imp_res = dp-sum(impulses.values())
    steps=[]; dofs=[]; muscles=[]
    for i in range(11):
        row=dict(arm=label,step=i,physical_s=float(t[i]),physical_ms=float(t[i]*1000),mass_native=M,
                 contact_Fz=float(external[i,2]),gravity_Fz=float(gravity[2]),applied_Fz=float(applied[i,2]),
                 load_ratio=float(external[i,2]/-gravity[2]),COM_az=float(comaz[i,2]),COM_vz=float(comv[i,2]),
                 root_qacc2=float(a['qacc'][i,2]),root_vz=float(a['qvel'][i,2]),
                 ctrl_max=float(a['ctrl'][i].max()),ctrl_at_one_count=int((a['ctrl'][i]==1).sum()),act_max=float(a['act'][i].max()),
                 active_force_L1=float(abs(active[i]).sum()),passive_muscle_force_L1=float(abs(passive[i]).sum()),
                 contact_rows=counts[i][0],noncontact_rows=counts[i][1],all48_closure_maxabs=float(abs(residual[i]).max()),
                 COM_force_residual_z=float((M*comaz-external-gravity-applied)[i,2]),COM_dp_z=float(dp[i]),
                 trapezoid_impulse_residual_z=float(imp_res[i]))
        row.update({f'power_{k}':float(v[i]) for k,v in power.items()})
        row.update({f'work_{k}':float(v[i]) for k,v in works.items()})
        row.update({f'impulse_{k}_z':float(v[i]) for k,v in impulses.items()})
        row.update({f'root2_{k}':float(v[i,2]) for k,v in terms.items()})
        steps.append(row)
        for d in range(48):
            dofs.append(dict(arm=label,step=i,physical_s=float(t[i]),dof=d,qvel=float(a['qvel'][i,d]),qacc=float(a['qacc'][i,d]),Ma=float(ma[i,d]),RHS=float(rhs[i,d]),residual=float(residual[i,d]),**{k:float(v[i,d]) for k,v in terms.items()}))
        for j in range(90):
            muscles.append(dict(arm=label,step=i,physical_s=float(t[i]),actuator=j,name=mm['actuator_names'][j],ctrl=float(a['ctrl'][i,j]),act=float(a['act'][i,j]),force_total=float(total[i,j]),force_active=float(active[i,j]),force_passive_bias=float(passive[i,j]),velocity=float(a['actuator_velocity'][i,j]),power_active=float(active[i,j]*a['actuator_velocity'][i,j]),power_passive_bias=float(passive[i,j]*a['actuator_velocity'][i,j])))
    summary=dict(errors_maxabs=errors,mass_native=M,work_native={k:float(v[-1]) for k,v in works.items()},
                 impulse_z_native={k:float(v[-1]) for k,v in impulses.items()},COM_dp_z=float(dp[-1]),trapezoid_impulse_residual_z=float(imp_res[-1]),
                 first_positive_sample_ms={k:next((float(t[i]*1000) for i,x in enumerate(v) if x>1e-10),None) for k,v in dict(COM_az=comaz[:,2],load_above_one=external[:,2]/-gravity[2]-1,activation=a['act'][:11].max(1),active_force_L1=abs(active).sum(1),active_power=power['muscle_active']).items()},
                 initial=steps[0],at_25us=steps[1],at_250us=steps[-1],
                 dynprm_unique=np.unique(m['actuator_dynprm'],axis=0).tolist(),actuator_dyntype_unique=np.unique(m['actuator_dyntype']).tolist(),
                 qM_projection='NOT_COMPUTED: packed qM saved but no dof_parentid or mass CSR layout saved in these inputs; Ma already saved and closed; no invented unpacking or COM internal-force contribution')
    return steps,dofs,muscles,summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    out=parser.parse_args().output or BASE/'s147-early-rebound-budget'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True,exist_ok=False)
    inputs=[Path(__file__),PROJECT/'verify_muscle_reconstruction.py',PROJECT/'analyze_free_physical_work.py',XML]
    inputs += [p/name for p in SOURCES.values() for name in FILES]
    before={str(p):sha(p) for p in inputs}
    dump(out/'INPUT_HASHES.json',before)
    dump(out/'PREANALYSIS.json',dict(task='S147/P1 early rebound diagnostic only, not product or walking acceptance',sources={k:str(v) for k,v in SOURCES.items()},samples=list(range(11)),physical_dt_s=25e-6,force_tolerance=1e-10,
         claims=['Scalar gain*act versus passive muscle bias, separate qfrc_passive; raw CSR projection and actual efc type/id/address classification.',
                 'All48 instantaneous Ma equation is not COM force attribution. Internal muscle can have root DOF2 component without external net COM force.',
                 'COM external contact+gravity+applied force balance; trapezoidal signed work/impulse are quadrature, not exact integrals.',
                 'Temporal compatibility is not unique causation; contact work does not establish stored contact elastic energy.',
                 'Native force/work/impulse physical units NOT_ESTABLISHED. No model/forward/step/neural runs or tuning.']))
    nodes=list(ET.parse(XML).getroot().find('actuator'));assert len(nodes)==90
    steps=[];dofs=[];muscles=[];summary={}
    for label,p in SOURCES.items():
        s,d,m,report=analyze(label,p,nodes);steps+=s;dofs+=d;muscles+=m;summary[label]=report
    table(out/'EARLY_STEPS.csv',steps);table(out/'GENERALIZED_ALL48.csv',dofs);table(out/'MUSCLE_SCALAR_SPLIT.csv',muscles)
    fig,axes=plt.subplots(3,2,figsize=(12,11),sharex=True)
    for label in SOURCES:
        s=[x for x in steps if x['arm']==label];t=[x['physical_ms'] for x in s]
        for ax,key,title in zip(axes.flat,['load_ratio','COM_az','COM_vz','active_force_L1','work_muscle_active','act_max'],['External contact load / weight','True COM az (native)','True COM vz (native)','Active actuator force L1 (native)','Signed active muscle work (native)','Real activation max; dashed ctrl max']):
            ax.plot(t,[x[key] for x in s],marker='.',label=label);ax.set_title(title);ax.grid(alpha=.3)
            if key=='act_max':ax.plot(t,[x['ctrl_max'] for x in s],linestyle='--',alpha=.6,label=label+' ctrl')
    for ax in axes[-1]:ax.set_xlabel('Actual physical time (ms)')
    for ax in axes.flat:ax.legend(fontsize=8)
    fig.suptitle('S147/P1 saved arrays only: 0–0.25 ms; temporal evidence, not unique causation\nNative physical force/work units NOT_ESTABLISHED')
    fig.tight_layout();fig.savefig(out/'EARLY_REBOUND_BUDGET.png',dpi=160);plt.close(fig)
    after={str(p):sha(p) for p in inputs};assert before==after
    dump(out/'SUMMARY.json',dict(passed=True,arms=summary,inputs_unchanged=True,model_load=0,forward=0,step=0,neural_runs=0))
    text='# S147/P1 early rebound saved-array diagnostic\n\n'
    text+='Actual steps 0..10, 25 us, ending 0.25 ms. No physics/neural rerun; no tuning. Native physical units **NOT_ESTABLISHED**.\n\n'
    for label,v in summary.items():
        text+=f'## {label}\n\n'
        for key in ('initial','at_25us','at_250us'):
            s=v[key];text+=f"- {key}: load={s['load_ratio']:.12g}, COMaz={s['COM_az']:.12g}, COMvz={s['COM_vz']:.12g}, ctrlmax={s['ctrl_max']:.12g}, ctrl==1 count={s['ctrl_at_one_count']}, actmax={s['act_max']:.12g}.\n"
        text+=f"- Signed work: `{json.dumps(v['work_native'])}`. COM dpz={v['COM_dp_z']:.12g}; trapezoidal impulse residual={v['trapezoid_impulse_residual_z']:.12g} (not exact integral).\n"
        text+=f"- First strictly positive sampled times (threshold 1e-10, ms): `{json.dumps(v['first_positive_sample_ms'])}`.\n"
        text+=f"- Native dynprm unique={v['dynprm_unique']}; dyntype={v['actuator_dyntype_unique']}. Fast submillisecond activation is compatible with saved .0005/.0004 s constants; no tuning or biological calibration claim.\n\n"
    text+='## Interpretation and limits\n\n'
    text+='Contact load/COMaz rise and activation/active muscle work establish concurrently at 25us sampling; nonzero initial actuator force includes passive muscle bias even at zero activation. This is temporally compatible with coupled muscle-state/contact dynamics, not proof of unique active-actuator causation or stored soft-contact elastic-energy release. No counterfactual was run.\n\n'
    text+='All48 generalized Ma balance is instantaneous. Root DOF2 is not true COM acceleration/force attribution: internal muscle generalized forces can have a root component yet have no net external COM force. True whole-body COM uses mass-weighted saved body Jacobian/Jacdot and external contact/gravity/applied world forces. Muscle-induced contact changes cannot be separated causally by this algebra. qfrc_passive is joint/other passive, not passive muscle bias. Signed work is generalized force dot qvel, not COM work.\n\n'
    text+=summary['S142']['qM_projection']+'.\n\nAll outputs are diagnostic, not stable support, physical prewarm or autonomous walking success. Inputs and source SHA256 are frozen; no tracker or historical artifact modified.\n'
    with (out/'REPORT.md').open('x',encoding='utf-8') as f:f.write(text)
    dump(out/'OUTPUT_HASHES.json',{str(p):sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(dict(READY=True,output=str(out),summary=str(out/'SUMMARY.json'),source_sha256=sha(__file__)),ensure_ascii=False))

if __name__=='__main__':
    main()
