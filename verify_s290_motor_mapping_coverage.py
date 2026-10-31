"""Independent S290 saved-data coverage verification; never loads a simulator/tier.

Historical signed moments are state-specific S84 evidence, not S289 torques.
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
ARCH=Path('/home/sensen/Desktop/engine修复验收')
REF=ARCH/'s289-decoder-range-support/20261026T173400Z/decoder_original'
XML=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
NAMES=ARCH/'s244-chirality-of-the-5-of-6-arm/20261025T205824474993Z/CMD_A3_sym_matched.npz'
GAIN=ARCH/'s255-static-gain-can-it-sustain/20261025T221308522972Z/ARM_VECTORS.json'
POOL=ROOT/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json'
MECH=ARCH/'s84-mechanical-mapping/20261023T201631Z'
WINDOWS={'early_0_150ms':(0.,150.),'precontact20ms':(233.4,253.4),'late350_400ms':(350.,400.)}

def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def expanded(path,stack=(),loader=None):
    path=Path(path).resolve()
    if path in stack:raise ValueError('include cycle')
    node=ET.fromstring(loader(path) if loader else path.read_bytes())
    def walk(parent):
        for child in list(parent):
            if child.tag=='include':
                if not child.get('file'):raise ValueError('missing include file')
                sub=expanded(path.parent/child.get('file'),stack+(path,),loader)
                index=list(parent).index(child);parent.remove(child)
                for offset,item in enumerate(sub):parent.insert(index+offset,copy.deepcopy(item))
            else:walk(child)
    walk(node);return node

def ordered(a,b,c):
    if any(len(x)!=90 or len(set(x))!=90 for x in (a,b,c)) or list(a)!=list(b) or list(a)!=list(c):raise ValueError('ordered unique90 contract')
    return True

def duration(mask):
    mask=np.asarray(mask,bool);edges=np.diff(np.r_[False,mask,False].astype(int));spans=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
    longest=float(spans.max()*.1) if spans.size else 0.
    return dict(duration_ms=float(mask.sum()*.1),longest_ms=longest,long_ge10ms=longest>=10.)

def equal(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,(float,np.floating)):return isinstance(b,(int,float)) and abs(float(a)-float(b))<=1e-10
    return a==b

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);out=parser.parse_args(argv).output
    checks=[]
    def check(name,value,**detail):checks.append(dict(name=name,passed=bool(value),**detail))
    try:
        pre=js(out/'PREREG.json');analysis=js(out/'ANALYSIS.json');rows=js(out/'PER_CHANNEL.json');cover=js(out/'COVERAGE_GROUPS.json')
        check('round S290',pre['round']=='S290')
        for p,h in pre['input_hashes'].items():check('input hash '+p,Path(p).is_file() and sha(p)==h)
        check('no FAILED marker',not (out/'FAILED.json').exists())
        for label,ledger in [('prereg',pre.get('calls',{})),('analysis',analysis.get('calls',{}))]:
            check(label+' all recorded calls zero',bool(ledger) and all(v==0 for v in ledger.values()))
        with np.load(NAMES,allow_pickle=True) as z:names=[str(v) for v in z['names']]
        tree=expanded(XML);act=[a for s in tree.findall('./actuator') for a in s];xmlnames=[a.get('name') for a in act]
        check('own recursive XML ordered contract',ordered(names,xmlnames,js(REF/'MODEL_METADATA.json')['names']))
        for label,a in [('swap',[names[1],names[0]]+names[2:]),('duplicate',names[:-1]+[names[0]]),('missing',names[:-1])]:
            rejected=False
            try:ordered(a,xmlnames,names)
            except ValueError:rejected=True
            check('own synthetic '+label+' rejected',rejected)
        files={Path('/s290/root.xml'):b'<mujoco><actuator><general name="a"/><include file="sub/a.xml"/><general name="d"/></actuator></mujoco>',Path('/s290/sub/a.xml'):b'<mujocoinclude><general name="b"/><include file="b.xml"/></mujocoinclude>',Path('/s290/sub/b.xml'):b'<mujocoinclude><general name="c"/></mujocoinclude>'}
        test=expanded('/s290/root.xml',loader=lambda p:files[p]);check('own nested relative include retains order',[e.get('name') for e in test.find('actuator')]==list('abcd'))
        reg=js(out/'CONTRACT_REGRESSION.json');check('analyzer synthetic cases actually present',all(any(t.get('case')==case and t.get('passed') for t in reg['tests']) for case in ('swapped','duplicate','missing','include_relative_inplace_nested','include_cycle')))
        ne=arrays(REF/'NEURAL.npz');ctrl=arrays(REF/'CTRL.npz')['ctrl'];g=np.asarray(js(GAIN)['v1']);pool=js(POOL)['measured'];resolved={r['actuator']:r for r in pool['resolved']};unresolved={r['actuator']:r for r in pool['unresolved']};source={**resolved,**unresolved}
        check('actual dimensions',ctrl.shape==(4000,90) and ne['raw_rates_hz'].shape==(400,2,90) and ne['rates_hz'].shape==(400,90))
        check('saved encoded/submitted/background real current identity',np.array_equal(ne['encoded_current'],ne['sensory_current']) and np.array_equal(ne['submitted_current'],ne['encoded_current']) and np.array_equal(ne['actual_step_current'],ne['background_current'][None,:]+np.repeat(ne['submitted_current'],2,axis=0)))
        for label,(start,end) in WINDOWS.items():
            sm=(np.arange(400)>=start-1e-9)&(np.arange(400)<end-1e-9)
            check('actual current finite window '+label,np.isfinite(ne['encoded_current'][sm]).all() and np.isfinite(ne['actual_step_current']).all(),encoded_min=float(ne['encoded_current'][sm].min()),encoded_max=float(ne['encoded_current'][sm].max()))
        check('own rate/gain decoder values',np.array_equal(ne['decoder_preclip'],ne['rates_hz']/450.*g) and np.array_equal(ne['decoder_postclip'],np.clip(ne['decoder_preclip'],0,1)))
        check('90 ordered output rows',len(rows)==90 and [r['name'] for r in rows]==names and [r['index'] for r in rows]==list(range(90)))
        check('gain vector actual archive identity',np.array_equal(g,ne['decoder_motor_gains']))
        check('disabled26 unresolved16 resolvedzero10',np.count_nonzero(g==0)==26 and len(unresolved)==16 and sum(g[i]==0 and n in resolved for i,n in enumerate(names))==10)
        check('unresolved all disabled',all(g[names.index(n)]==0 for n in unresolved))
        check('pool source complete',len(source)==90 and set(source)==set(names))
        rawtime=np.arange(800)*.5+.5;ct=np.arange(4000)*.1;recomputed=[]
        tendon={e.get('name'):e for s in tree.findall('./tendon') for e in s}
        for i,n in enumerate(names):
            row=rows[i];r=source[n];roots=sorted(set(map(int,r.get('pool_root_ids',[])))) if n in resolved else []
            check('channel '+str(i)+' own mapping/gain',row['pool_status']==('resolved_declared' if n in resolved else 'unresolved_declared') and row['gain']==g[i] and row['zero_gain']==(g[i]==0) and row['pool_root_ids']==roots and row['leg']==r['leg'])
            check('channel '+str(i)+' source annotation',equal(row['pool_source_annotation'],r))
            check('channel '+str(i)+' local identity boundary',row['local_indices'] is None and str(row['local_index_boundary_check']).startswith('NOT_ESTABLISHED') and row['tier_root_order_status']=='NOT_ESTABLISHED')
            t=tendon.get(act[i].get('tendon'));fixed={e.get('joint'):float(e.get('coef','1')) for e in t.findall('./joint')} if t is not None and t.tag=='fixed' else {}
            check('channel '+str(i)+' tendon declaration',row['tendon']==act[i].get('tendon') and row['tendon_type']==(t.tag if t is not None else 'NOT_ESTABLISHED') and equal(row['fixed_joint_coefficients'],fixed))
            wins={}
            for label,(start,end) in WINDOWS.items():
                rm=(rawtime>=start-1e-9)&(rawtime<end-1e-9);cm=(ct>=start-1e-9)&(ct<end-1e-9);raw=ne['raw_rates_hz'][:,:,i].reshape(800)[rm];active=bool(np.any(raw>0));v=dict(raw_active=active,mapped_no_observed_spikes=n in resolved and not active,disabled_by_zero_gain=bool(g[i]==0),mapping_unresolved=n not in resolved,high_applied=duration(ctrl[cm,i]>=.8),mean_applied_command=float(ctrl[cm,i].mean()),raw_rate_max_hz=float(raw.max()));wins[label]=v
                check('channel '+str(i)+' window '+label,equal(v,row['windows'][label]))
            recomputed.append(dict(leg=r['leg'],windows=wins))
        for leg in ('LF','RF','LM','RM','LH','RH'):
            rr=[r for r in recomputed if r['leg']==leg]
            for label in WINDOWS:
                rrw=[r['windows'][label] for r in rr];expected=dict(channels=len(rr),raw_active=sum(r['raw_active'] for r in rrw),mapped_no_observed_spikes=sum(r['mapped_no_observed_spikes'] for r in rrw),disabled_by_zero_gain=sum(r['disabled_by_zero_gain'] for r in rrw),mapping_unresolved=sum(r['mapping_unresolved'] for r in rrw),high_long_ge10ms=sum(r['high_applied']['long_ge10ms'] for r in rrw));check('own leg coverage '+leg+'/'+label,equal(expected,cover['coverage6legs'][leg][label]))
        check('summary gain/map counts',analysis['zero_gain_count']==26 and analysis['resolved_declared']==74 and analysis['unresolved_declared']==16)
        mechpre=js(MECH/'PREREG.json');mapping=list(csv.DictReader((MECH/'ACTUATOR_MAPPING.csv').open()));check('historical XML identity',mechpre['identity'][str(XML)]==sha(XML));check('historical compiled actuator order',len(mapping)==90 and [r['actuator'] for r in mapping]==names and [int(r['index']) for r in mapping]==list(range(90)))
        moments={s:arrays(MECH/(s+'_MECHANICS.npz'))['moment'] for s in ('initial','first_activation','formal_start')};check('historic moment dimensions',all(v.shape==(90,48) and np.isfinite(v).all() for v in moments.values()))
        jointrows=list(csv.DictReader((MECH/'STATE_JOINT_FORCES.csv').open()));jointnames={int(r['dof']):r['joint'] for r in jointrows if r['state']=='initial'};check('historical DOF name coverage',set(jointnames)==set(range(48)))
        expected_dofs=[j for j in range(48) if np.any(moments['initial'][:,j]>1e-12) and np.any(moments['initial'][:,j]<-1e-12)]
        signed=cover['historical_signed_moment_groups'];check('historical signed group DOF completeness',[r['dof'] for r in signed]==expected_dofs and cover['historical_signed_group_count']==len(expected_dofs) and analysis['historical_signed_moment_groups']==len(expected_dofs))
        for group in signed:
            j=group['dof'];sides={s:dict(positive_indices=np.flatnonzero(m[:,j]>1e-12).tolist(),negative_indices=np.flatnonzero(m[:,j]<-1e-12).tolist()) for s,m in moments.items()};p=sides['initial']['positive_indices'];n=sides['initial']['negative_indices']
            check('signed DOF '+str(j)+' source sides',equal(group['state_sides'],sides) and group['positive_indices']==p and group['negative_indices']==n and group['joint']==jointnames[j])
            check('signed DOF '+str(j)+' boundary and sign stability',group['current_S289_moment_signs']=='NOT_ESTABLISHED' and group['sign_groups_identical_across_three_saved_states']==all(v==sides['initial'] for v in sides.values()))
            for label,(start,end) in WINDOWS.items():
                mask=(ct>=start-1e-9)&(ct<end-1e-9);positive=np.any(ctrl[mask][:,p]>=.8,axis=1);negative=np.any(ctrl[mask][:,n]>=.8,axis=1);expected=dict(positive_side_only_high=duration(positive&~negative),negative_side_only_high=duration(negative&~positive),both_sides_high=duration(positive&negative),neither_side_high=duration(~positive&~negative))
                check('signed DOF '+str(j)+' own command dwell '+label,equal(expected,group['windows'][label]))
        for i,row in enumerate(rows):
            expected={s:[dict(dof=int(j),joint=jointnames[int(j)],coefficient=float(m[i,j])) for j in np.flatnonzero(abs(m[i])>1e-12)] for s,m in moments.items()};check('channel '+str(i)+' own historic signed moments',equal(expected,row['historical_moment_signs']))
        check('historical order claim grounded',cover['historical_compiled_order_same_xml'] is True and analysis['historical_compiled_order_same_xml'] is True)
        notes=ROOT/'s290_provenance_notes.json'
        check('provenance notes supplied',notes.is_file())
        if notes.is_file():
            note=js(notes);check('provenance notes hashed',str(notes) in pre['input_hashes'] and pre['input_hashes'][str(notes)]==sha(notes))
            asset=note['actual_assets'];up=np.asarray(js(asset['upstream_gains'])['perrow_apex'])
            with np.load(asset['calibration_source_rates'],allow_pickle=False) as z:old=z['ctrl'][17].copy()
            with np.load(asset['replay_target_and_order'],allow_pickle=False) as z:target=z['cmd'].astype(float)
            expected=np.zeros(90);valid=(old>0)&(target>0);expected[valid]=target[valid]/old[valid]
            check('independent S254 calibration formula',np.array_equal(up,expected))
            check('independent S255 inherited gain formula',np.array_equal(g,1.236819199*up))
            check('independent zero provenance24 replayzeros2 silentcalibration',int(np.sum((g==0)&(target==0)))==24 and np.flatnonzero((g==0)&(target>0)).tolist()==[55,62])
            zero=note['per_zero_actuator'];check('provenance all zero rows exactly',[r['index'] for r in zero]==np.flatnonzero(g==0).tolist())
            for r in zero:
                i=r['index'];name=names[i];reason='inherited_replay_target_zero' if target[i]==0 else 'positive_replay_target_but_silent_old_calibration_source'
                check('zero provenance '+str(i),r['name']==name and r['gain']==0 and r['mapping']==('resolved' if name in resolved else 'unresolved') and r['zero_reason']==reason)
                if name in resolved:check('resolved zero real raw activity '+str(i),float(ne['raw_rates_hz'][:,:,i].max())==r['S289_raw_rate_max_hz'] and abs(float(ne['rates_hz'][:,i].max())-r['S289_filtered_rate_max_hz'])<1e-10 and not np.any(ctrl[:,i]))
            check('archive provenance copy',equal(js(out/'PROVENANCE.json'),note))
        check('analysis claims success',analysis.get('passed') is True)
    except Exception as e:check('verification exception',False,error=repr(e))
    result=dict(passed=all(c['passed'] for c in checks),checks=checks,failed=[c for c in checks if not c['passed']],verifier_sha256=sha(__file__),calls=dict(model_load=0,forward=0,physics_step=0,neural_tick=0,tier_build=0),limitations=['Historical S84 moments do not establish current S289 moment arms or torques','No selected tier root order is reconstructed or invented'])
    (out/'VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(passed=result['passed'],checks=len(checks),failed=result['failed'])));return 0 if result['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
