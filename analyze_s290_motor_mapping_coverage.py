"""S290 offline motor mapping/coverage audit; no simulator or tier imports.

Only reads historical artifacts. Declaration order is not claimed as observed
compiled order. Spatial tendon names do not establish antagonism or torque.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
ARCHIVE=Path('/home/sensen/Desktop/engine修复验收')
REF=ARCHIVE/'s289-decoder-range-support/20261026T173400Z/decoder_original'
FROZEN=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
NAMES=ARCHIVE/'s244-chirality-of-the-5-of-6-arm/20261025T205824474993Z/CMD_A3_sym_matched.npz'
GAINS=ARCHIVE/'s255-static-gain-can-it-sustain/20261025T221308522972Z/ARM_VECTORS.json'
POOLS=ROOT/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json'
MOTOR=ROOT/'engine/embodied/motor_output.py'
RUNNER=ROOT/'run_s289_decoder_range_support.py'
LEGS=('LF','RF','LM','RM','LH','RH')
WINDOWS={'early_0_150ms':(0.,150.),'precontact20ms':(233.4,253.4),'late350_400ms':(350.,400.)}
HIGH=.8
LONG_MS=10.
MECH=ARCHIVE/'s84-mechanical-mapping/20261023T201631Z'
MECH_STATES=('initial','first_activation','formal_start')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(str(type(v)))
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=serial)+'\n',encoding='utf-8')


def expand_xml(path, loader=None, stack=(), observed=None):
    """Expand each include in-place, relative to THAT including file, detecting cycles.

    MuJoCo include root is a wrapper; its children replace the include node.
    loader is injectable for in-memory synthetic include/cycle tests.
    """
    path=Path(path).resolve()
    if path in stack:raise ValueError('XML include cycle: '+' -> '.join(map(str,(*stack,path))))
    if observed is not None:observed.add(path)
    root=ET.fromstring(loader(path) if loader else path.read_bytes())
    def visit(node):
        for child in list(node):
            if child.tag=='include':
                rel=child.get('file')
                if not rel:raise ValueError('include missing file')
                sub=expand_xml(path.parent/rel,loader,(*stack,path),observed)
                at=list(node).index(child);node.remove(child)
                for offset,item in enumerate(list(sub)):
                    node.insert(at+offset,copy.deepcopy(item))
            else:visit(child)
    visit(root)
    return root


def declarations(tree):
    return [e for section in tree.findall('./actuator') for e in list(section)]


def name_contract(asset,xml_names,metadata,nu):
    for label,names in [('asset',asset),('xml',xml_names),('metadata_asset_copy',metadata)]:
        if len(names)!=90 or len(set(names))!=90:raise ValueError(label+' requires90 unique names')
    if nu!=90 or list(asset)!=list(xml_names) or list(asset)!=list(metadata):
        raise ValueError('ordered90 declaration/asset/archive contract mismatch')
    return True


def index_contract(rows,n):
    for ai,row in enumerate(rows):
        if row['actuator_index']!=ai:raise ValueError('actuator index/order mismatch')
        ids=row['local_indices']
        if len(set(ids))!=len(ids) or any(isinstance(i,bool) or not isinstance(i,int) or i<0 or i>=n for i in ids):raise ValueError('invalid local indices')
    return True


def regress(names):
    tests=[]
    for case,rows in [('negative_index',[{'actuator_index':0,'local_indices':[-1]}]),('upper_index',[{'actuator_index':0,'local_indices':[2]}]),('duplicate_index',[{'actuator_index':0,'local_indices':[1,1]}]),('wrong_actuator_index',[{'actuator_index':1,'local_indices':[0]}])]:
        rejected=False
        try:index_contract(rows,2)
        except ValueError:rejected=True
        tests.append({'case':case,'passed':rejected,'synthetic_only_actual_runtime_indices_NOT_ESTABLISHED':True})
    for label,mut in [('swapped',names[1:2]+names[:1]+names[2:]),('duplicate',[names[0]]+names[:-1]),('missing',names[:-1])]:
        rejected=False
        try:name_contract(mut,names,names,90)
        except ValueError:rejected=True
        tests.append({'case':label,'expected_reject':True,'passed':rejected})
    files={Path('/synthetic/main.xml'):b'<mujoco><actuator><general name="a"/><include file="sub/part.xml"/><general name="d"/></actuator></mujoco>',Path('/synthetic/sub/part.xml'):b'<mujocoinclude><general name="b"/><include file="nested.xml"/></mujocoinclude>',Path('/synthetic/sub/nested.xml'):b'<mujocoinclude><general name="c"/></mujocoinclude>'}
    tree=expand_xml('/synthetic/main.xml',loader=lambda p:files[p]);tests.append({'case':'include_relative_inplace_nested','passed':[x.get('name') for x in declarations(tree)]==list('abcd')})
    files[Path('/synthetic/sub/nested.xml')]=b'<mujocoinclude><include file="../main.xml"/></mujocoinclude>'
    rejected=False
    try:expand_xml('/synthetic/main.xml',loader=lambda p:files[p])
    except ValueError:rejected=True
    tests.append({'case':'include_cycle','passed':rejected})
    return {'passed':all(t['passed'] for t in tests),'tests':tests,'synthetic_only':True}


def longest(mask,dt=.1):
    padded=np.r_[False,np.asarray(mask,bool),False].astype(int);d=np.diff(padded)
    spans=np.flatnonzero(d==-1)-np.flatnonzero(d==1)
    return float(spans.max()*dt) if spans.size else 0.


def dwell(mask):
    longest_ms=longest(mask)
    return {'duration_ms':float(np.count_nonzero(mask)*.1),'longest_ms':longest_ms,'long_ge10ms':longest_ms>=LONG_MS}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    out=ap.parse_args(argv).output
    if not out.is_absolute() or out.exists():raise ValueError('fresh absolute --output required')
    includes=set();tree=expand_xml(FROZEN,observed=includes)
    inputs=[Path(__file__).resolve(),RUNNER,MOTOR,ROOT/'engine/embodied/adapters.py',FROZEN,NAMES,GAINS,POOLS,REF/'NEURAL.npz',REF/'CTRL.npz',REF/'MODEL_METADATA.json',REF/'COMMAND_TIMING.npz',REF.parent/'PREREG.json',REF.parent/'METRICS.json',*sorted(includes)]
    inputs += [MECH/'PREREG.json',MECH/'audit.py',MECH/'ACTUATOR_MAPPING.csv',MECH/'STATE_JOINT_FORCES.csv',*[MECH/(s+'_MECHANICS.npz') for s in MECH_STATES]]
    notes_path=ROOT/'s290_provenance_notes.json';notes=js(notes_path)
    inputs += [notes_path,*[Path(p) for p in notes['actual_assets'].values()],*[Path(r['path']) for r in notes['loading_chain']]]
    hashes={str(p):sha(p) for p in dict.fromkeys(inputs)}
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'PREREG.json',{'round':'S290','mode':'saved-data/read-only simulator; no simulator/network imports or tier builds','input_hashes':hashes,'windows_ms':WINDOWS,'window_semantics':'half-open applied intervals; late400 boundary contains no additional application step','first_nonfoot_ms_reference':253.4,'high_command_threshold':HIGH,'threshold_interpretation':'explicitly descriptive, not torque/causal/anatomical evidence','long_duration_ms':LONG_MS,'calls':{'model_load':0,'forward':0,'physics_step':0,'neural_tick':0,'tier_build':0},'compiled_actuator_order':'NOT_ESTABLISHED; metadata names are a loaded asset copy, not mj_id2name observations','local_root_mapping':'NOT_ESTABLISHED unless exact saved selected-tier root order exists; do not build tier','antagonist_rule':'fixed exact opposite joints plus historical S84 SAME-XML signed moment groups per DOF, abs(moment)>1e-12; no S289 state-sign/net-torque claim','historical_mechanical_reference':'S84 initial/first_activation/formal_start; historical compiled order observed in mapping CSV/audit, not a new S289 mj_id2name observation'})
    (out/Path(__file__).name).write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    try:
        with np.load(NAMES,allow_pickle=True) as z:names=[str(x) for x in z['names']]
        with np.load(REF/'NEURAL.npz',allow_pickle=False) as z:ne={k:z[k] for k in ('raw_rates_hz','rates_hz','decoder_motor_gains')}
        with np.load(REF/'CTRL.npz',allow_pickle=False) as z:ctrl=z['ctrl'].copy()
        meta=js(REF/'MODEL_METADATA.json');act=declarations(tree);xml_names=[e.get('name') for e in act]
        name_contract(names,xml_names,meta['names'],ctrl.shape[1]);regression=regress(names);dump(out/'CONTRACT_REGRESSION.json',regression)
        if not regression['passed']:raise ValueError('synthetic contract regression failed')
        assert ctrl.shape==(4000,90) and ne['raw_rates_hz'].shape==(400,2,90) and ne['rates_hz'].shape==(400,90)
        gains=np.asarray(js(GAINS)['v1'],float);assert gains.shape==(90,) and np.array_equal(gains,ne['decoder_motor_gains'])
        source=js(POOLS)['measured'];resolved=source['resolved'];unresolved=source['unresolved'];records=resolved+unresolved
        assert len(records)==90 and len({r['actuator'] for r in records})==90 and {r['actuator'] for r in records}==set(names)
        by_name={r['actuator']:r for r in records};resolved_names={r['actuator'] for r in resolved}
        tendons={e.get('name'):e for section in tree.findall('./tendon') for e in section if e.get('name')}
        parents={child:node for node in tree.iter() for child in node}
        site_paths={}
        for site in tree.findall('.//worldbody//site'):
            chain=[];node=parents.get(site)
            while node is not None:
                if node.tag=='body':chain.append({'body':node.get('name'),'joints':[j.get('name') for j in node.findall('./joint')]})
                node=parents.get(node)
            site_paths[site.get('name')]=chain
        rows=[];raw_time=np.arange(800)*.5+.5;applied_time=np.arange(4000)*.1
        for i,(name,a) in enumerate(zip(names,act)):
            r=by_name[name];t=tendons.get(a.get('tendon'));kind=t.tag if t is not None else 'NOT_ESTABLISHED'
            joint_coeffs={e.get('joint'):float(e.get('coef','1')) for e in t.findall('./joint')} if t is not None and kind=='fixed' else {}
            roots=sorted(set(int(v) for v in r.get('pool_root_ids',[]))) if name in resolved_names else []
            row={'index':i,'name':name,'leg':r.get('leg','NOT_ESTABLISHED'),'side':r.get('side','NOT_ESTABLISHED'),'tendon':a.get('tendon'),'tendon_type':kind,'fixed_joint_coefficients':joint_coeffs,'joint_sign_evidence':'XML fixed coefficients' if joint_coeffs else 'NOT_ESTABLISHED for current S289 state: spatial signs not archived; historical S84 reference moments saved separately','pool_status':'resolved_declared' if name in resolved_names else 'unresolved_declared','pool_root_ids':roots,'pool_source_annotation':r,'tier_root_order_status':'NOT_ESTABLISHED','local_indices':None,'local_index_boundary_check':'NOT_ESTABLISHED: selected tier root order/manifest not archived','gain':float(gains[i]),'zero_gain':bool(gains[i]==0),'gain_provenance':'declared ARM_VECTORS v1; biological calibration NOT_ESTABLISHED','windows':{}}
            for label,(start,end) in WINDOWS.items():
                rm=(raw_time>=start-1e-9)&(raw_time<end-1e-9);cm=(applied_time>=start-1e-9)&(applied_time<end-1e-9)
                active=bool(np.any(ne['raw_rates_hz'][:,:,i].reshape(800)[rm]>0))
                row['windows'][label]={'raw_active':active,'mapped_no_observed_spikes':bool(name in resolved_names and not active),'disabled_by_zero_gain':bool(gains[i]==0),'mapping_unresolved':name not in resolved_names,'high_applied':dwell(ctrl[cm,i]>=HIGH),'mean_applied_command':float(ctrl[cm,i].mean()),'raw_rate_max_hz':float(ne['raw_rates_hz'][:,:,i].reshape(800)[rm].max())}
            row['spatial_sites_to_body_ancestor_joints']=[{'site':e.get('site'),'ancestor_body_joints':site_paths.get(e.get('site'),[])} for e in t.findall('./site')] if t is not None and kind=='spatial' else []
            row['joints']=sorted({j for item in row['spatial_sites_to_body_ancestor_joints'] for body in item['ancestor_body_joints'] for j in body['joints']}) if kind=='spatial' else sorted(joint_coeffs)
            row['joint_list_caveat']='Ancestor-chain association only; not proof of nonzero moment arm or sign' if kind=='spatial' else 'fixed signed coefficients'
            rows.append(row)
        groups=[]
        for i in range(90):
            x=rows[i]['fixed_joint_coefficients']
            if not x:continue
            for j in range(i+1,90):
                y=rows[j]['fixed_joint_coefficients']
                if not y or x.keys()!=y.keys() or not all(x[k]*y[k]<0 for k in x):continue
                group={'indices':[i,j],'names':[names[i],names[j]],'joint_set':sorted(x),'evidence':'opposite fixed tendon coefficients on exact same joints; not biological antagonist validation','windows':{}}
                for label,(start,end) in WINDOWS.items():
                    mask=(applied_time>=start-1e-9)&(applied_time<end-1e-9);a=ctrl[mask,i]>=HIGH;b=ctrl[mask,j]>=HIGH
                    group['windows'][label]={'only_first_high':dwell(a&~b),'only_second_high':dwell(b&~a),'both_high':dwell(a&b),'neither_high':dwell(~a&~b)}
                groups.append(group)
        # Historical SAME-XML signed moment rows provide grounded reference-state
        # opposing-side groups. These are not current S289 Jacobians or net torques.
        assert js(MECH/'PREREG.json')['identity'][str(FROZEN)]==sha(FROZEN)
        with (MECH/'ACTUATOR_MAPPING.csv').open(encoding='utf-8',newline='') as f:historical_map=list(csv.DictReader(f))
        assert [r['actuator'] for r in historical_map]==names
        assert [int(r['index']) for r in historical_map]==list(range(90))
        moments={}
        for state in MECH_STATES:
            with np.load(MECH/(state+'_MECHANICS.npz'),allow_pickle=False) as z:moments[state]=z['moment'].copy()
            assert moments[state].shape==(90,48)
        with (MECH/'STATE_JOINT_FORCES.csv').open(encoding='utf-8',newline='') as f:jr=list(csv.DictReader(f))
        dof_names={int(r['dof']):r['joint'] for r in jr if r['state']=='initial'}
        historical_groups=[]
        for dof in range(48):
            col=moments['initial'][:,dof];plus=np.flatnonzero(col>1e-12).tolist();minus=np.flatnonzero(col< -1e-12).tolist()
            if not plus or not minus:continue
            state_sides={state:{'positive_indices':np.flatnonzero(m[:,dof]>1e-12).tolist(),'negative_indices':np.flatnonzero(m[:,dof]< -1e-12).tolist()} for state,m in moments.items()}
            group={'dof':dof,'joint':dof_names.get(dof,'NOT_ESTABLISHED'),'positive_indices':plus,'negative_indices':minus,'historical_reference':'S84 initial same XML, archived actuator_moment coefficient signs; pulling-force sign may invert both sides, not opposing relation','state_sides':state_sides,'sign_groups_identical_across_three_saved_states':all(v['positive_indices']==plus and v['negative_indices']==minus for v in state_sides.values()),'current_S289_moment_signs':'NOT_ESTABLISHED','windows':{}}
            for label,(start,end) in WINDOWS.items():
                mask=(applied_time>=start-1e-9)&(applied_time<end-1e-9);p=np.any(ctrl[mask][:,plus]>=HIGH,axis=1);n=np.any(ctrl[mask][:,minus]>=HIGH,axis=1)
                group['windows'][label]={'positive_side_only_high':dwell(p&~n),'negative_side_only_high':dwell(n&~p),'both_sides_high':dwell(p&n),'neither_side_high':dwell(~p&~n)}
            historical_groups.append(group)
        for row in rows:
            i=row['index'];row['historical_moment_signs']={state:[{'dof':int(d),'joint':dof_names.get(int(d)),'coefficient':float(m[i,d])} for d in np.flatnonzero(abs(m[i])>1e-12)] for state,m in moments.items()}
            row['historical_compiled_name_order']='S84 mj_id2name-derived ACTUATOR_MAPPING.csv exact asset order; same frozen XML'
        coverage={}
        for leg in LEGS:
            rr=[r for r in rows if r['leg']==leg];assert len(rr)==15
            coverage[leg]={label:{'channels':15,'raw_active':sum(r['windows'][label]['raw_active'] for r in rr),'mapped_no_observed_spikes':sum(r['windows'][label]['mapped_no_observed_spikes'] for r in rr),'disabled_by_zero_gain':sum(r['zero_gain'] for r in rr),'mapping_unresolved':sum(r['pool_status']=='unresolved_declared' for r in rr),'high_long_ge10ms':sum(r['windows'][label]['high_applied']['long_ge10ms'] for r in rr)} for label in WINDOWS}
        zero_notes={r['index']:r for r in notes['per_zero_actuator']}
        for row in rows:
            z=zero_notes.get(row['index']);row['zero_reason']=z['zero_reason'] if z else 'gain_nonzero'
            row['zero_reason_evidence']=z
            row['disablement_interpretation']='Inherited declared engineering mask, biological disablement NOT_ESTABLISHED' if z else 'not fixed-disabled'
            row['evidence_category']='no_mapping_in_current_annotation' if row['pool_status']=='unresolved_declared' else ('declared_historical_mask_with_active_current_pool' if z else 'declared_mapping_and_nonzero_gain')
        dump(out/'PROVENANCE.json',notes)
        dump(out/'PER_CHANNEL.json',rows)
        with (out/'PER_CHANNEL.csv').open('w',encoding='utf-8',newline='') as f:
            fields=['index','name','leg','tendon','tendon_type','joints','fixed_joint_coefficients','spatial_sites_to_body_ancestor_joints','pool_status','pool_root_ids','gain','zero_gain','zero_reason','evidence_category','disablement_interpretation','zero_reason_evidence','local_index_boundary_check','windows'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for r in rows:writer.writerow({k:json.dumps(r[k],ensure_ascii=False,default=serial) if isinstance(r[k],(list,dict)) else r[k] for k in fields})
        dump(out/'COVERAGE_GROUPS.json',{'coverage6legs':coverage,'opposite_fixed_coefficient_groups':groups,'spatial_antagonism':'S84 reference-state opposite moment signs grounded below; actual S289 current signed moment arms and biological antagonism NOT_ESTABLISHED; no name-only assertions','group_count':len(groups),'historical_signed_moment_groups':historical_groups,'historical_signed_group_count':len(historical_groups),'historical_compiled_order_same_xml':True,'overlap_warning':'raw-active, disabled and mapped-no-spikes categories are explicit overlapping descriptors, not exclusive classes'})
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(6,1,figsize=(18,22),sharex=True)
        for ax,leg in zip(axes,LEGS):
            rr=[r for r in rows if r['leg']==leg];idx=[r['index'] for r in rr];im=ax.imshow(ctrl[:,idx].T,origin='upper',aspect='auto',extent=[0,400,14.5,-.5],vmin=0,vmax=1,cmap='viridis');ax.set_yticks(range(15),[f"{r['index']:02d} {'ZERO ' if r['zero_gain'] else ''}{r['name']}" for r in rr],fontsize=7);ax.set_title(leg+' applied command; ZERO = declared zero V1 gain');ax.axvline(150,color='white',ls=':');ax.axvline(233.4,color='orange',ls=':');ax.axvline(253.4,color='red',ls='--');ax.axvline(350,color='white',ls=':')
        axes[-1].set_xlabel('actual application time ms');fig.suptitle('S290 saved command coverage — no torque, antagonism or walking success inference');fig.tight_layout(rect=[0,0,1,.98]);fig.savefig(out/'COVERAGE.png',dpi=140);plt.close(fig)
        unchanged=all(sha(p)==h for p,h in hashes.items());assert unchanged
        dump(out/'ANALYSIS.json',{'passed':True,'declaration_asset_order_contract':True,'compiled_order':'NOT_ESTABLISHED','n_channels':90,'n_legs':6,'zero_gain_count':int(np.count_nonzero(gains==0)),'resolved_declared':len(resolved),'unresolved_declared':len(unresolved),'local_index_boundary_check':'NOT_ESTABLISHED','root_to_local_semantics':'build_motor_manifest sorted unique source roots then retain only roots in exact tier_root_ids order; runtime selected order absent; no tier build','source_validator_caveat':'build_motor_manifest enforces name set, not ordered physical actuator names; this offline contract newly enforces XML declaration order','fixed_opposite_groups':len(groups),'historical_signed_moment_groups':len(historical_groups),'historical_compiled_order_same_xml':True,'spatial_signs':'S84 three reference-state signed moments saved; current S289 signs NOT_ESTABLISHED','input_unchanged':unchanged,'calls':{'model_load':0,'forward':0,'physics_step':0,'neural_tick':0,'tier_build':0},'limitations':['Coverage describes declared mapping and saved rates/commands only','High >=.8 and long>=10ms are descriptive thresholds, not torque or causal evidence','Observed compiled actuator order and actual local indices remain NOT_ESTABLISHED']})
        print(json.dumps({'passed':True,'channels':90,'groups':len(groups)}));return 0
    except Exception as e:
        dump(out/'FAILED.json',{'error':repr(e),'calls':{'model_load':0,'forward':0,'physics_step':0,'neural_tick':0,'tier_build':0}})
        raise

if __name__=='__main__':raise SystemExit(main())
