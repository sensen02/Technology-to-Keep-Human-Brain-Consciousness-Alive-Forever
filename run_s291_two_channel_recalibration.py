"""S291/P3: original mask versus externally derived gains55/62 only.

Import safe. No historical runner is imported or executed. Requires a passed,
explicit derivation; never selects a calibration sample or invents a gain.
Original must exactly reproduce S289 decoder_original before candidate loading.
"""
from pathlib import Path
import argparse
import csv
import dataclasses
import hashlib
import json
import traceback
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
ARCHIVE=Path('/home/sensen/Desktop/engine修复验收')
S289=ROOT/'run_s289_decoder_range_support.py'
REFERENCE=ARCHIVE/'s289-decoder-range-support/20261026T173400Z'
FROZEN=Path('/run/media/sensen/Data2/engine修复验收/s27-graded-dose/20261023T035715Z/MODEL_CALIB_H_VERIFIED.xml')
INITIAL=ARCHIVE/'s177-root-newton-search/20261025T020500000000Z/ROOT_BALANCE_STATE.npz'
COMMAND_NAMES=ARCHIVE/'s244-chirality-of-the-5-of-6-arm/20261025T205824474993Z/CMD_A3_sym_matched.npz'
GAINS=ARCHIVE/'s255-static-gain-can-it-sustain/20261025T221308522972Z/ARM_VECTORS.json'
POOLS=ROOT/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json'
MOTOR_SOURCE=ROOT/'engine/embodied/motor_output.py'
DERIVATION=ROOT/'s291_calibration_derivation.json'
LEGS=('lf','lm','lh','rf','rm','rh')
ARMS=('original_mask','recalibrated_55_62')
SENSORY_GAINS={a:1. for a in ARMS}
DECODER_DENOMINATORS={a:450. for a in ARMS}
CHANGED={55:'RMF_accesory_trochanter_flexor',62:'LHC_pleural_remotor_and_abductor'}
TARGET='RHC_pleural_remotor_and_abductor'
W=24.446354755878
BUDGET=dict(model_load=2,tier_build=1,network_init=2,live_step=8000,neural_tick=1600,live_fwd_position=2,live_forward=0,diagnostic_forward=8002)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(str(type(value)))
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=serial)+'\n',encoding='utf-8')
def exact_array(x,y):
    if x.shape!=y.shape or x.dtype!=y.dtype:return False
    return bool(np.array_equal(x,y,equal_nan=True)) if np.issubdtype(x.dtype,np.inexact) else bool(np.array_equal(x,y))


def timing_regression():
    result={}
    for arm in ARMS:
        rows=[]
        for i in range(400):
            source=i-1;valid_from_ms=float(source+1) if source>=0 else 0.
            rows.append(dict(application_interval=i,source_interval=source,valid_from_ms=valid_from_ms,applied_start_ms=float(i),applied_end_ms=float(i+1),anticipatory=bool(i<valid_from_ms)))
        violations=sum(r['anticipatory'] for r in rows)
        result[arm]=dict(violations=violations,expected_violations=0,passed=violations==0,intervals=rows,final_decoded=dict(source_interval=399,source_start_ms=399.,source_end_ms=400.,valid_from_ms=400.,actual_applied_start_ms=None,actual_applied_end_ms=None,planned_start_ms=400.,planned_end_ms=401.,pending=True))
    result['passed']=all(result[a]['passed'] for a in ARMS)
    return result


def compare_archive(actual,reference):
    checks=[]
    for name in ('TRAJECTORY.npz','NEURAL.npz','CTRL.npz','COMMAND_TIMING.npz','INITIAL_CARRY.npz'):
        with np.load(actual/name,allow_pickle=False) as got,np.load(reference/name,allow_pickle=False) as ref:
            missing=sorted(set(ref.files)-set(got.files));checks.append(dict(file=name,field='reference_fields_present',passed=not missing,missing=missing))
            for key in ref.files:
                if key not in got.files:continue
                x,y=got[key],ref[key];checks.append(dict(file=name,field=key,passed=exact_array(x,y),actual_shape=list(x.shape),reference_shape=list(y.shape),actual_dtype=str(x.dtype),reference_dtype=str(y.dtype)))
    for name in ('STATE.csv','CHANNELS.json','CONTACTS.json','PARAMETERS.json','DECODED_COMMANDS.json','COMMAND_APPLICATIONS.json'):
        checks.append(dict(file=name,field='file_bytes',passed=(actual/name).read_bytes()==(reference/name).read_bytes()))
    passed=all(r['passed'] for r in checks)
    return dict(passed=passed,method='all reference NPZ fields exact dtype/shape/value with matching NaNs; CSV/JSON bytes exact',checks=checks,reference=str(reference),candidate_allowed=passed)


def read_derivation(path,old,names):
    """Strict externally supplied values; no implicit calibration or defaults."""
    derivation=json.loads(path.read_text())
    if derivation.get('passed') is not True:raise ValueError('derivation must explicitly have passed=true')
    entries=derivation.get('per_channel')
    if isinstance(entries,dict):
        entries=[dict(v,index=int(k)) if isinstance(v,dict) and 'index' not in v else v for k,v in entries.items()]
    if not isinstance(entries,list):raise ValueError('derivation.per_channel must list exactly indices55,62')
    parsed={}
    new=old.copy()
    for row in entries:
        index=row.get('index')
        if isinstance(index,bool) or not isinstance(index,int) or index not in CHANGED or index in parsed:raise ValueError('derivation indices must be unique55,62')
        if names[index]!=CHANGED[index] or row.get('actuator',row.get('name'))!=names[index]:raise ValueError('derivation actuator identity mismatch')
        gain=row.get('gain_new',row.get('new_gain'))
        if isinstance(gain,bool) or not isinstance(gain,(int,float)) or not np.isfinite(gain) or gain<=0:raise ValueError('derived gain must be explicitly finite positive')
        if old[index]!=0:raise ValueError('expected original zero masks55,62')
        new[index]=float(gain);parsed[index]=row
    if set(parsed)!=set(CHANGED):raise ValueError('derivation must supply only both55,62')
    if np.flatnonzero(new!=old).tolist()!=list(CHANGED):raise ValueError('only gains55,62 may change')
    return derivation,parsed,new


def runtime_manifest_check(man,roots,names,pools,derived_rows):
    """Validate actual runtime projection; fail closed before neural/physics ticks."""
    if roots.ndim!=1 or roots.dtype.kind not in 'iu' or len(set(map(int,roots)))!=len(roots):raise ValueError('tier roots must be unique integer vector')
    if man['n_neurons']!=len(roots) or man['n_actuators']!=90 or len(man['rows'])!=90:raise ValueError('manifest dimension mismatch')
    source={r['actuator']:r for r in pools['measured']['resolved']}
    checks=[]
    for ai,row in enumerate(man['rows']):
        if isinstance(row['actuator_index'],(bool,np.bool_)) or not isinstance(row['actuator_index'],(int,np.integer)) or row['actuator_index']!=ai or row['actuator']!=names[ai]:raise ValueError('manifest actuator order mismatch')
        local=row['local_indices']
        if any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) or i<0 or i>=len(roots) for i in local) or len(set(map(int,local)))!=len(local):raise ValueError('manifest local index type/uniqueness/bounds')
        projected=[int(roots[i]) for i in local]
        if projected!=list(map(int,row.get('tier_root_ids',[]))):raise ValueError('local indices do not recover actual tier roots')
        if not set(projected)<=set(map(int,row.get('pool_root_ids',[]))):raise ValueError('projected roots not source pool')
        checks.append(dict(index=ai,actuator=names[ai],status=row['status'],local_indices=list(map(int,local)),projected_roots=projected,passed=True))
    target_checks=[]
    for ai,name in CHANGED.items():
        row=man['rows'][ai]
        if name not in source:raise ValueError('target pool unresolved')
        expected=sorted(set(map(int,source[name]['pool_root_ids'])))
        if not expected or len(expected)!=len(source[name]['pool_root_ids']):raise ValueError('target source roots must be nonempty unique')
        if row['status']!='complete' or row.get('missing_tier_roots') or list(map(int,row['pool_root_ids']))!=expected or list(map(int,row['tier_root_ids']))!=expected:raise ValueError('target actual tier coverage must be complete; partial/absent forbidden')
        if any(int(np.count_nonzero(roots==r))!=1 for r in expected):raise ValueError('target root must occur exactly once in actual tier')
        declared=derived_rows[ai].get('pool_root_ids')
        if declared is not None and sorted(map(int,declared))!=expected:raise ValueError('derivation source root identity mismatch')
        target_checks.append(dict(index=ai,actuator=name,source_pool_root_ids=expected,local_indices=row['local_indices'],projected_roots=[int(roots[i]) for i in row['local_indices']],status=row['status'],passed=True))
    return dict(passed=True,n_neurons=len(roots),n_actuators=90,all_rows=checks,recalibrated_targets=target_checks,checked_before='any model forward/fwdPosition, network init, neural ticks or physics steps')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--derivation',type=Path,default=DERIVATION)
    args=parser.parse_args(argv);out=args.output;derivation_path=args.derivation.resolve()
    if not out.is_absolute() or out.exists():raise ValueError('fresh absolute --output required')
    out=out.resolve()
    xml_tree=ET.fromstring(FROZEN.read_bytes())
    if xml_tree.findall('.//include'):raise ValueError('frozen XML unexpectedly contains include; declaration parser must be extended before running')
    xml_names=[e.get('name') for sec in xml_tree.findall('./actuator') for e in sec]
    with np.load(COMMAND_NAMES,allow_pickle=True) as z:names=[str(x) for x in z['names']]
    if len(names)!=90 or len(set(names))!=90 or names!=xml_names:raise ValueError('90-name asset/XML declaration order mismatch')
    old_gains=np.asarray(json.loads(GAINS.read_text())['v1'],float)
    if old_gains.shape!=(90,) or np.count_nonzero(old_gains==0)!=26:raise ValueError('original V1 mismatch')
    # A failed/missing derivation is a STOP with no imports or dynamic calls.
    try:derivation,derived_rows,new_gains=read_derivation(derivation_path,old_gains,names)
    except Exception:
        rejection_traceback=traceback.format_exc()
        rejected_paths=[Path(__file__).resolve(),S289,FROZEN,INITIAL,COMMAND_NAMES,GAINS,POOLS,MOTOR_SOURCE,REFERENCE/'PREREG.json',REFERENCE/'MANIFEST_SHA256.json',*sorted((ROOT/'engine').rglob('*.py'))]
        rejected_paths += [Path(e.attrib['file']) for e in xml_tree.findall('.//asset/mesh')]
        rejected_paths += sorted(p for p in (REFERENCE/'decoder_original').iterdir() if p.is_file())
        derivation_snapshot=None;registered_evidence=[]
        if derivation_path.is_file():
            rejected_paths.append(derivation_path)
            try:derivation_snapshot=json.loads(derivation_path.read_text())
            except (ValueError,UnicodeError):derivation_snapshot={'parseable':False,'raw_text':derivation_path.read_text(errors='replace')}
        notes_path=ROOT/'s290_provenance_notes.json'
        evidence_documents=[('derivation',derivation_snapshot)]
        if notes_path.is_file():
            rejected_paths.append(notes_path)
            try:evidence_documents.append(('s290_notes',json.loads(notes_path.read_text())))
            except (ValueError,UnicodeError):pass
        for label,document in evidence_documents:
            if not isinstance(document,dict):continue
            for field in ('source_hashes','input_hashes'):
                registered=document.get(field,{})
                if isinstance(registered,dict):
                    for p,expected in registered.items():
                        path=Path(p);exists=path.is_file();actual=sha(path) if exists else None
                        registered_evidence.append(dict(document=label,field=field,path=str(path),registered_sha256=expected,actual_sha256=actual,present=exists,hash_matches=actual==expected))
                        if exists:rejected_paths.append(path)
            assets=document.get('actual_assets',{})
            if isinstance(assets,dict):
                for p in assets.values():
                    if isinstance(p,str) and Path(p).is_file():rejected_paths.append(Path(p))
            for row in document.get('loading_chain',[]):
                if isinstance(row,dict) and isinstance(row.get('path'),str) and Path(row['path']).is_file():rejected_paths.append(Path(row['path']))
        prereg_stop=dict(round='S291',arms=list(ARMS),duration_ms=400,derivation=str(derivation_path),derivation_sha256=sha(derivation_path) if derivation_path.is_file() else None,derivation_passed=False,source_hashes={str(p):sha(p) for p in dict.fromkeys(rejected_paths)},registered_evidence_hash_checks=registered_evidence,source_sha256=sha(__file__),original_xml=str(FROZEN),original_xml_sha256=sha(FROZEN),reference_arm=str(REFERENCE/'decoder_original'),dynamic_calls_allowed=False,planned_only_gain_indices=[55,62],original_gains=old_gains.tolist(),new_gains=None,unknown_gain_values='NOT_ESTABLISHED; no substitutions or sample selection',planned_decoder_config=dict(rate_for_full_control_hz=450.,filter_tau_s=.005,max_control=1.,sensory_gain=1.),analysis_windows=dict(early_ms=[0.,150.],early_partition='before versus at/after own first nonfoot contact separate',own_precontact_ms=20.,own_precontact_policy='max(0,first-20)..first; disclose phase/truncation',late_ms=[350.,400.]),time_axes=dict(body_dt_s=.0001,neural_dt_s=.0005,sensory_dt_s=.001,causal_application='current[i,i+1]ms uses previous sourcei-1; initial zero carry; source399 valid400 pending not applied'),reason='No gains inferred from an unpassed derivation; no model/neural imports permitted')
        out.mkdir(parents=True,exist_ok=False)
        dump(out/'PREREG.json',prereg_stop)
        if derivation_snapshot is not None:dump(out/'DERIVATION.json',derivation_snapshot)
        dump(out/'CONFIG_DIFF.json',dict(planned_only_change_indices=[55,62],old_gains=old_gains.tolist(),new_gains=None,derivation_passed=False,executed=False))
        dump(out/'CALL_LEDGER.json',{k:0 for k in BUDGET})
        dump(out/'STOP.json',dict(reason='derivation rejected or unavailable',traceback=rejection_traceback,candidate_executed=False,original_executed=False,calls={k:0 for k in BUDGET}))
        return 2
    ref_prereg=json.loads((REFERENCE/'PREREG.json').read_text());ref_manifest=json.loads((REFERENCE/'MANIFEST_SHA256.json').read_text())
    assert sha(FROZEN)==ref_prereg['original_xml_sha256']
    assert sha(S289)==ref_prereg['source_hashes'][str(S289)]==ref_manifest[str(S289)]
    assert all(sha(p)==h for p,h in ref_prereg['source_hashes'].items()),'S289 runtime/model dependency changed'
    assert str(REFERENCE/'MANIFEST_SHA256.json') not in ref_manifest
    assert all(sha(p)==h for p,h in ref_manifest.items()),'S289 sealed archive changed'
    paths=[Path(__file__).resolve(),S289,FROZEN,INITIAL,COMMAND_NAMES,GAINS,POOLS,derivation_path,ROOT/'data/flywire/banc_connectome.npz',ROOT/'outputs/neural_rebuild/sensory_channel_lookup.json',*sorted((ROOT/'engine').rglob('*.py')),REFERENCE/'PREREG.json',REFERENCE/'MANIFEST_SHA256.json']
    paths += [Path(e.attrib['file']) for e in xml_tree.findall('.//asset/mesh')]
    paths += sorted(p for p in (REFERENCE/'decoder_original').iterdir() if p.is_file())
    # Any externally registered derivation inputs are also frozen, not interpreted as code.
    for field in ('source_hashes','input_hashes'):
        registered=derivation.get(field,{})
        if isinstance(registered,dict):
            for p,h in registered.items():
                assert sha(p)==h,'derivation registered input changed';paths.append(Path(p))
    hashes={str(p):sha(p) for p in dict.fromkeys(paths)}
    motor_lines=MOTOR_SOURCE.read_text().splitlines()
    formula_lines=[dict(line=i+1,text=t) for i,t in enumerate(motor_lines) if 151<=i+1<=172]
    assert 'rate = self.rates / self.config.rate_for_full_control_hz' in '\n'.join(motor_lines)
    assert 'rate = rate * gvec' in '\n'.join(motor_lines) and 'ctrl = np.clip(rate, 0, self.config.max_control)' in '\n'.join(motor_lines)
    arm_gains={ARMS[0]:old_gains,ARMS[1]:new_gains}
    decoder_configs={a:dict(rate_for_full_control_hz=450.,filter_tau_s=.005,max_control=1.,per_actuator_gain=arm_gains[a].tolist()) for a in ARMS}
    prereg=dict(round='S291',goal='P3 source-derived two-channel translation recalibration sensitivity, not behavior acceptance',arms=list(ARMS),duration_ms=400,seed=0,network='all_empty ASSUMED_NETWORK_SCORE',fingerprint='56e0322a33c51b7f03aaae9f7df1ef3f',exc_tau_ms=3.,inh_tau_ms=3.,dt_s=.0001,neural_dt_ms=.5,sensory_ms=1,source_hashes=hashes,original_xml=str(FROZEN),original_xml_sha256=sha(FROZEN),historical_s289_sha256=sha(S289),reference_arm=str(REFERENCE/'decoder_original'),derivation=str(derivation_path),derivation_sha256=sha(derivation_path),derivation_passed=True,derived_channels=derived_rows,model_diff='none: original complete XML and original friction',sensory_gains=SENSORY_GAINS,decoder_configs=decoder_configs,parameter_diff={'only_change':'per_actuator_gain indices55,62 from original0 to explicitly supplied passed derivation gains; denominator450, filter5ms, sensorygain1, background, other88 gains, neural/physics/causal unchanged','changed_indices':[55,62],'old':decoder_configs[ARMS[0]],'new':decoder_configs[ARMS[1]]},decoder_formula={'preclip':'(own rates_hz /450)*own per_actuator_gain','postclip':'np.clip(preclip,0,1)','filter':'per-neural-sample exp(-.0005/.005) recurrence','actual_source':str(MOTOR_SOURCE),'actual_source_sha256':sha(MOTOR_SOURCE),'actual_source_lines':formula_lines},runtime_prerequisites={'tier':'save exact actual roots and manifest; target source root sets unique/nonempty, fully present exactly once, all local identities/int uniqueness/bounds validated before any network/fwd/step','model':'each compiled90 actuator names from mj_id2name must exactly equal asset and XML declaration before own fwdPosition/network; candidate model only after original exact gate'},initial_carry={'values':[0.]*90,'valid_from_s':0.,'source_interval':-1,'source_start_s':None,'source_end_s':None,'causal_application_s':[0.,.001]},time_axes={'body_state_s':'0..0.4 dt.0001','encoder_s':'i*.001, i0..399','neural_s':'i*.001+[.0005,.001]','decoded_command_s':'source[i,i+1]ms valid_from=i+1ms','application_s':'current[i,i+1] uses source i-1 initial-1','last_command':'source399 valid400ms pending, not applied','ctrl_time_s':'historical step END, separate application start/end'},analysis_windows={'early_ms':[0.,150.],'early_partition':'before vs at/after first nonfoot contact separate','own_precontact_ms':20.,'own_precontact_policy':'max(0,first-20)..first; disclose differing phases/truncation','late_ms':[350.,400.],'late_policy':'inclusive state end400, no additional application','metrics':['foot/nonfoot native Fz','COMz/dz/vz','rootdz','orientation','limits','target spikes/state','90 raw/filter/preclip/postclip/applied','bound dwell']},current_recording={'encoded_current':'400,N unchanged encoder','submitted_current':'encoded*1 BOTH','background_current':'same unchanged N','actual_step_current':'real net.step input argument copied before original unchanged call800,N'},decoder_recording={'decoder_preclip':'400,90 own rates/450*own gain','decoder_postclip':'returned carry400,90','decoder_motor_gains':'own90','decoder_rate_for_full_control_hz':'450 scalar','filter_tau_s':'.005 scalar','decoder_max_control':'1 scalar'},gate='original_mask all S289 decoder_original shared NPZ fields exact including gain/current/timing plus JSON/CSV; fail before candidate model/network',budget=BUDGET,budget_on_gate_fail='model1/tier1/net1/step4000/tick800/fwdPosition1/diagForward4001',regression='both pure causal0',sync='unchanged copyData+diagnostic forward no diagnostic feedback',sequence='set ctrlcarry; applied copy; encode; advance2; own decode; oldcarry10steps;snapshot',legacy_force='frame@raw up=-z unchanged live',diagnostic_force='floor-side-sign frame.T@raw',contact_raw_columns=['geom1','geom2','dist','pos_x','pos_y','pos_z',*['frame_'+str(i) for i in range(9)],*['force_'+str(i) for i in range(6)]],body_velocity='world angular xyz then linear xyz BODY flg_local0',limitations='Only declared two-channel interface sensitivity; biological calibration/physical units/autonomous support/walking NOT_ESTABLISHED',non_neural_controller=False,auxiliary_force=False,extra_stimulation=False,training=False,prewarm=False)
    out.mkdir(parents=True,exist_ok=False);dump(out/'PREREG.json',prereg)
    dump(out/'DERIVATION.json',derivation);dump(out/'CONFIG_DIFF.json',prereg['parameter_diff'])
    regression=timing_regression();dump(out/'TIMESTAMP_REGRESSION.json',regression)
    calls={k:0 for k in BUDGET};dump(out/'CALL_LEDGER.json',calls)
    if not regression['passed']:
        dump(out/'STOP.json',dict(reason='pure timing regression failed',candidate_executed=False,calls=calls));return 2
    try:
        import mujoco as mj
        from engine.embodied.adapters import build_neural_tier,TierConfig,NeuralTier,OwnershipLedger
        from engine.embodied.sensory_channels import ChannelSensoryEncoder,ChannelQuantities,ChannelConfig,channel_index_map
        from engine.embodied.motor_output import build_motor_manifest,MotorOutput,MotorOutputConfig
        built=build_neural_tier(TierConfig(sign_fill='all_empty').validate());calls['tier_build']+=1;dump(out/'CALL_LEDGER.json',calls)
        assert built['fingerprint']==prereg['fingerprint']
        N=int(built['n']);roots=np.asarray(built['tier']['root_ids'])
        if roots.ndim!=1 or roots.dtype.kind not in 'iu' or any(int(r)<0 for r in roots):
            dump(out/'STOP.json',dict(reason='actual tier roots invalid unsigned identity vector',candidate_executed=False,calls=calls));return 2
        roots=roots.astype(np.uint64,copy=True)
        np.savez_compressed(out/'TIER_ROOT_IDS.npz',root_ids=roots.copy())
        pools=json.loads(POOLS.read_text())
        try:
            man=build_motor_manifest(pools,roots,names);dump(out/'MOTOR_MANIFEST.json',man)
            runtime_check=runtime_manifest_check(man,roots,names,pools,derived_rows)
        except Exception:
            dump(out/'RUNTIME_MAPPING_CHECK.json',dict(passed=False,error=traceback.format_exc()))
            dump(out/'STOP.json',dict(reason='runtime tier mapping prerequisite failed before neural/physics dynamics',candidate_executed=False,calls=calls));return 2
        dump(out/'RUNTIME_MAPPING_CHECK.json',runtime_check)
        groups,_=channel_index_map(N,roots)
        with np.load(INITIAL) as z:qrel=np.asarray(z['qpos'],float).copy()
        resolved={r['actuator']:r for r in pools['measured']['resolved']}
        cells=[int(np.flatnonzero(roots==int(c))[0]) for c in resolved[TARGET]['pool_root_ids'] if np.any(roots==int(c))]
        assert cells and len(names)==90
        initial_checks=[];parameter_reference=None;background_reference=None

        def run(label):
            nonlocal parameter_reference,background_reference
            assert label in ARMS
            gains=arm_gains[label].copy();sensory_gain=1.;decoder_denominator=450.
            dest=out/label;dest.mkdir()
            m=mj.MjModel.from_xml_path(str(FROZEN));calls['model_load']+=1;dump(out/'CALL_LEDGER.json',calls)
            compiled_names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)]
            compiled_ok=len(compiled_names)==90 and len(set(compiled_names))==90 and compiled_names==names==xml_names
            dump(dest/'COMPILED_MAPPING.json',dict(passed=compiled_ok,actuator_names=compiled_names,compiled_actuator_names=compiled_names,asset_names=names,xml_declaration_names=xml_names,xml_sha256=sha(FROZEN),source='actual mj_id2name on THIS loaded model before fwdPosition/network init',runtime_mapping_check_passed=runtime_check['passed'],calls_at_check=calls.copy()))
            if not compiled_ok:
                dump(out/'STOP.json',dict(reason='compiled model actuator order prerequisite failed',arm=label,candidate_executed=False,calls=calls));raise ValueError('compiled actuator identity/order mismatch')
            assert m.opt.timestep==.0001 and (m.nq,m.nv,m.nu,m.nmesh)==(49,48,90,71)
            assert m.npair==0
            geom_names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g) for g in range(m.ngeom)]
            foot_names={lg.upper()+'Tarsus'+str(i)+'_geom' for lg in LEGS for i in range(1,6)}
            foot={g:geom_names[g][:2].lower() for g in range(m.ngeom) if geom_names[g] in foot_names};assert len(foot)==30
            floor=geom_names.index('floor');assert np.array_equal(m.geom_friction,np.tile([1.,.005,.0001],(m.ngeom,1)));assert np.all(m.geom_priority==0)
            joints={}
            for lg in LEGS:
                ids=[j for j in range(m.njnt) if (mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j) or '').lower().startswith('joint_'+lg)]
                tib=[j for j in ids if 'tibia' in mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j).lower()][0]
                joints[lg]=dict(tib_q=int(m.jnt_qposadr[tib]),tib_v=int(m.jnt_dofadr[tib]),q=[int(m.jnt_qposadr[j]) for j in ids])
            tib0=np.array([qrel[joints[lg]['tib_q']] for lg in LEGS])
            dump(dest/'MODEL_METADATA.json',dict(names=names,body_names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,i) for i in range(m.nbody)],geom_names=geom_names,geom_leg=foot,floor=floor,geom_bodyid=m.geom_bodyid,geom_friction=m.geom_friction,geom_priority=m.geom_priority,geom_condim=m.geom_condim,integrator=int(m.opt.integrator),timestep=m.opt.timestep,nmesh=m.nmesh,target_index=names.index(TARGET),target_cells=cells,fingerprint=built['fingerprint'],model_xml_sha256=sha(FROZEN)))
            d=mj.MjData(m);d.qpos[:]=qrel;d.qvel[:]=0;d.act[:]=0;d.ctrl[:]=0
            mj.mj_fwdPosition(m,d);calls['live_fwd_position']+=1
            q0=d.qpos.copy();diag=mj.MjData(m)
            tier=NeuralTier(built,TierConfig(sign_fill='all_empty',background_seed=0).validate(),OwnershipLedger());calls['network_init']+=1
            tier.net.p=dataclasses.replace(tier.net.p,inh_tau_ms=tier.net.p.exc_tau_ms);tier.net.p.validate();parameters=dataclasses.asdict(tier.net.p)
            assert parameters['inh_tau_ms']==parameters['exc_tau_ms']==3.
            if parameter_reference is None:parameter_reference=parameters
            else:assert parameters==parameter_reference
            background_current=np.array(tier.background_nA,dtype=float,copy=True);assert background_current.shape==(N,) and np.isfinite(background_current).all()
            if background_reference is None:background_reference=background_current.copy()
            else:assert np.array_equal(background_reference,background_current)
            dump(dest/'PARAMETERS.json',parameters)
            dump(dest/'SENSORY_GAIN_CONFIG.json',dict(arm=label,sensory_gain=sensory_gain,submitted_formula='encoded_current * gain',clipping=False,signed=True,background_unchanged=True))
            initial_checks.append(dict(arm=label,qpos_sha256=hashlib.sha256(d.qpos.tobytes()).hexdigest(),qvel_sha256=hashlib.sha256(d.qvel.tobytes()).hexdigest(),act_sha256=hashlib.sha256(d.act.tobytes()).hexdigest(),ctrl_sha256=hashlib.sha256(d.ctrl.tobytes()).hexdigest(),seed=0,fingerprint=built['fingerprint'],background_sha256=hashlib.sha256(background_current.tobytes()).hexdigest()))
            enc=ChannelSensoryEncoder(groups,N,ChannelConfig());dec=MotorOutput(man,MotorOutputConfig(rate_for_full_control_hz=decoder_denominator,per_actuator_gain=gains,filter_tau_s=5e-3))
            dump(dest/'DECODER_CONFIG.json',dict(arm=label,**dataclasses.asdict(dec.config),sensory_gain=sensory_gain,formula='(rates_hz / rate_for_full_control_hz) * own gains then clip[0,max_control]',fixed_zero_gain_channels=int(np.count_nonzero(gains==0)),source=str(MOTOR_SOURCE),source_sha256=sha(MOTOR_SOURCE),formula_source_lines=formula_lines,derivation_sha256=sha(derivation_path),changed_gain_indices=[] if label==ARMS[0] else [55,62]))
            carry=np.zeros(90);np.savez_compressed(dest/'INITIAL_CARRY.npz',command=carry.copy(),valid_from_s=np.array(0.),source_interval=np.array(-1),source_start_s=np.array(np.nan),source_end_s=np.array(np.nan))
            traj,ctrls,rows,ticks,sens,rates,rawrates,channels=[],[],[],[],[],[],[],[]
            submitted_currents,actual_step_currents=[],[];decoder_preclips,decoder_postclips=[],[];decoded,applications=[],[]
            step=tier.net.step;expected_step_drive=None
            def tick(*a,**kw):
                if a:
                    assert 'current_nA' not in kw;actual_drive=np.array(a[0],dtype=float,copy=True)
                else:actual_drive=np.array(kw.get('current_nA',0.0),dtype=float,copy=True)
                assert actual_drive.shape==(N,) and np.isfinite(actual_drive).all() and expected_step_drive is not None
                assert np.array_equal(actual_drive,expected_step_drive);actual_step_currents.append(actual_drive)
                spikes=step(*a,**kw);calls['neural_tick']+=1
                ticks.append(dict(time_ms=tier.net.t_ms,spikes=spikes[cells].copy(),ge=tier.net.ge[cells].copy(),gi=tier.net.gi[cells].copy(),v=tier.net.v[cells].copy()))
                return spikes
            tier.net.step=tick
            def ledger(dd):
                lp=dict(lp=0.,lm=0.,bp=0.,bm=0.,np_=0,nm=0);per={lg:dict(touch=0.,F_world=np.zeros(3),F_up=0.) for lg in LEGS}
                for ci in range(dd.ncon):
                    c=dd.contact[ci];g1,g2=int(c.geom1),int(c.geom2)
                    if floor not in (g1,g2):continue
                    other=g2 if g1==floor else g1;fx=np.zeros(6);mj.mj_contactForce(m,dd,ci,fx)
                    fw=np.asarray(c.frame,float).reshape(3,3)@fx[:3];up=-float(fw[2])
                    if other in foot:
                        lg=foot[other];per[lg]['touch']=1.;per[lg]['F_world']+=fw;per[lg]['F_up']+=up;lp['lp' if c.dist<0 else 'lm']+=up
                    else:lp['bp' if c.dist<0 else 'bm']+=up;lp['np_' if c.dist<0 else 'nm']+=1
                return lp,per
            def snapshot():
                mj.mj_copyData(diag,m,d);mj.mj_forward(m,diag);calls['diagnostic_forward']+=1
                mass=m.body_mass;cv=np.zeros((m.nbody,6))
                for b in range(m.nbody):mj.mj_objectVelocity(m,diag,mj.mjtObj.mjOBJ_BODY,b,cv[b],0)
                force=np.zeros((6,3));penetrating=np.zeros(6,int);body=0;raw=[]
                for ci in range(diag.ncon):
                    c=diag.contact[ci];g1,g2=int(c.geom1),int(c.geom2);fx=np.zeros(6);mj.mj_contactForce(m,diag,ci,fx)
                    raw.append([g1,g2,float(c.dist),*c.pos,*np.asarray(c.frame).ravel(),*fx])
                    if floor not in (g1,g2):continue
                    other=g2 if g1==floor else g1;fw=(1 if g1==floor else -1)*(np.asarray(c.frame).reshape(3,3).T@fx[:3])
                    if other in foot:
                        li=LEGS.index(foot[other]);force[li]+=fw;penetrating[li]+=int(c.dist<0)
                    elif c.dist<0:body+=1
                limited=np.flatnonzero(m.jnt_limited);qa=m.jnt_qposadr[limited]
                limit=np.maximum(np.maximum(m.jnt_range[limited,0]-diag.qpos[qa],diag.qpos[qa]-m.jnt_range[limited,1]),0).max()
                return dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),act=d.act.copy(),ctrl=d.ctrl.copy(),qacc=diag.qacc.copy(),body_xpos=diag.xpos.copy(),body_xipos=diag.xipos.copy(),body_velocity=cv,com=(mass[:,None]*diag.xipos).sum(0)/mass.sum(),com_velocity=(mass[:,None]*cv[:,3:]).sum(0)/mass.sum(),foot_force_world=force,foot_penetrating_count=penetrating,body_penetrating_contacts=body,max_limit=limit,external_applied=diag.xfrc_applied.copy(),qfrc_applied=diag.qfrc_applied.copy(),contact_raw=raw,qfrc_actuator=diag.qfrc_actuator.copy(),qfrc_constraint=diag.qfrc_constraint.copy())
            traj.append(snapshot())
            for i in range(400):
                d.ctrl[:]=carry;applied=carry.copy();lp,per=ledger(d)
                qv={lg:ChannelQuantities(tibia_joint_dev_rad=float(d.qpos[joints[lg]['tib_q']]-q0[joints[lg]['tib_q']]),tibia_joint_vel_rad_s=float(d.qvel[joints[lg]['tib_v']]),multi_joint_dev_rad=float(np.mean(d.qpos[joints[lg]['q']]-q0[joints[lg]['q']])),contact_force_norm=float(np.linalg.norm(per[lg]['F_world'])),foot_contact=float(per[lg]['touch'])) for lg in LEGS}
                cur,rec=enc.encode(qv);sens.append(cur.copy());channels.append(rec);submitted=cur*sensory_gain;submitted_currents.append(submitted.copy())
                assert np.array_equal(tier.background_nA,background_current);expected_step_drive=tier.background_nA+submitted;before_steps=len(actual_step_currents)
                sp=tier.advance(2,submitted);assert len(actual_step_currents)==before_steps+2
                rawrates.append([np.array([sample[idx].mean()/.0005 if len(idx) else 0 for idx in dec.groups]) for sample in sp])
                carry,info=dec.consume(sp,spike_dt_s=.0005,start_s=i*.001,end_s=i*.001+.001,connected=True);rates.append(np.asarray(info['rates_hz']))
                preclip=(np.asarray(info['rates_hz'],dtype=float)/dec.config.rate_for_full_control_hz)*gains
                assert np.array_equal(carry,np.clip(preclip,0,dec.config.max_control));decoder_preclips.append(preclip.copy());decoder_postclips.append(carry.copy())
                a=np.clip(applied,0,1.);pending=i==399;planned_start=(i+True)*.001;planned_end=planned_start+.001
                decoded.append(dict(source_interval=i,source_start_s=info['source_start_s'],source_end_s=info['source_end_s'],valid_from_s=info['valid_from_s'],command=carry.copy(),planned_applied_start_s=planned_start,planned_applied_end_s=planned_end,actual_applied_start_s=None if pending else planned_start,actual_applied_end_s=None if pending else planned_end,pending_at_capture_end=pending))
                src=i-1;applications.append(dict(application_interval=i,applied_start_s=i*.001,applied_end_s=(i+1)*.001,source_interval=src,source_start_s=None if src<0 else src*.001,source_end_s=None if src<0 else src*.001+.001,valid_from_s=0. if src<0 else src*.001+.001,command=a.copy(),initial_carry=src<0,anticipatory=False))
                for j in range(10):
                    d.ctrl[:]=a;ctrls.append(a.copy());mj.mj_step(m,d);calls['live_step']+=1;traj.append(snapshot())
                lp,_=ledger(d);tib=np.array([d.qpos[joints[lg]['tib_q']] for lg in LEGS]);dx,dy=float(d.qpos[0]-q0[0]),float(d.qpos[1]-q0[1])
                rows.append(dict(t_ms=float(i+1),dx=dx,dy=dy,dz=float(d.qpos[2]-q0[2]),horiz=float(np.hypot(dx,dy)),leg_up_pen_W=lp['lp']/W,tib_excursion=float(np.mean(abs(tib-tib0))),n_body_ground=int(lp['np_']+lp['nm']),nbg_pen=int(lp['np_'])))
            counts=np.array([len(r['contact_raw']) for r in traj],dtype=np.int64);offsets=np.concatenate(([0],np.cumsum(counts)))
            raw=np.asarray([c for r in traj for c in r['contact_raw']],dtype=float).reshape(-1,21);arrays={k:np.asarray([r[k] for r in traj]) for k in traj[0] if k!='contact_raw'}
            np.savez_compressed(dest/'TRAJECTORY.npz',**arrays,contact_raw=raw,contact_offsets=offsets,contact_count=counts);dump(dest/'CONTACTS.json',[r['contact_raw'] for r in traj])
            encoded_array=np.asarray(sens);submitted_array=np.asarray(submitted_currents);actual_array=np.asarray(actual_step_currents)
            assert encoded_array.shape==submitted_array.shape==(400,N) and actual_array.shape==(800,N)
            assert np.array_equal(submitted_array,encoded_array*sensory_gain) and np.array_equal(actual_array,np.repeat(background_current[None,:]+submitted_array,2,axis=0)) and np.array_equal(tier.background_nA,background_current)
            assert np.asarray(decoder_preclips).shape==np.asarray(decoder_postclips).shape==(400,90)
            np.savez_compressed(dest/'NEURAL.npz',time_ms=np.array([r['time_ms'] for r in ticks]),**{k:np.array([r[k] for r in ticks]) for k in ('spikes','ge','gi','v')},target_cells=np.array(cells),sensory_current=encoded_array,encoded_current=encoded_array,submitted_current=submitted_array,background_current=background_current,actual_step_current=actual_array,sensory_time_ms=np.arange(400,dtype=float),actual_step_time_ms=np.array([r['time_ms'] for r in ticks]),rates_hz=np.asarray(rates),raw_rates_hz=np.asarray(rawrates),ctrl=np.asarray(ctrls),ctrl_time_s=np.asarray(arrays['time'][1:]),decoder_preclip=np.asarray(decoder_preclips),decoder_postclip=np.asarray(decoder_postclips),decoder_motor_gains=gains.copy(),decoder_rate_for_full_control_hz=np.array(dec.config.rate_for_full_control_hz),filter_tau_s=np.array(dec.config.filter_tau_s),decoder_max_control=np.array(dec.config.max_control))
            dump(dest/'CURRENT_PATH_CHECK.json',dict(passed=True,arm=label,sensory_gain=sensory_gain,encoded_shape=list(encoded_array.shape),submitted_shape=list(submitted_array.shape),actual_step_shape=list(actual_array.shape),background_shape=list(background_current.shape),actual_argument_capture='net.step positional current_nA (or keyword), copied before original bound step; exactly one original call',actual_equals_background_plus_submitted=True,background_retained=True,submitted_equals_encoded_times_gain=True,clipping=False,signed=True,calls_recorded=len(actual_step_currents)))
            np.savez_compressed(dest/'CTRL.npz',ctrl=np.asarray(ctrls),names=np.asarray(names));dump(dest/'CHANNELS.json',channels);dump(dest/'DECODED_COMMANDS.json',decoded);dump(dest/'COMMAND_APPLICATIONS.json',applications)
            np.savez_compressed(dest/'COMMAND_TIMING.npz',decoded=np.asarray([r['command'] for r in decoded]),applied=np.asarray([r['command'] for r in applications]),applied_source_index=np.asarray([r['source_interval'] for r in applications]),pending_final=carry.copy(),decoded_command=np.asarray([r['command'] for r in decoded]),source_start_s=np.asarray([r['source_start_s'] for r in decoded]),source_end_s=np.asarray([r['source_end_s'] for r in decoded]),valid_from_s=np.asarray([r['valid_from_s'] for r in decoded]),planned_applied_start_s=np.asarray([r['planned_applied_start_s'] for r in decoded]),planned_applied_end_s=np.asarray([r['planned_applied_end_s'] for r in decoded]),actual_applied_start_s=np.asarray([np.nan if r['actual_applied_start_s'] is None else r['actual_applied_start_s'] for r in decoded]),actual_applied_end_s=np.asarray([np.nan if r['actual_applied_end_s'] is None else r['actual_applied_end_s'] for r in decoded]),pending_at_capture_end=np.asarray([r['pending_at_capture_end'] for r in decoded]),applied_command=np.asarray([r['command'] for r in applications]),applied_start_s=np.asarray([r['applied_start_s'] for r in applications]),applied_end_s=np.asarray([r['applied_end_s'] for r in applications]),applied_source_interval=np.asarray([r['source_interval'] for r in applications]),applied_valid_from_s=np.asarray([r['valid_from_s'] for r in applications]),initial_carry=np.zeros(90))
            with (dest/'STATE.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            assert len(traj)==4001 and len(ctrls)==4000 and len(ticks)==800 and np.asarray(ctrls).shape==(4000,90)
            assert np.array_equal(arrays['qpos'][0],qrel) and not np.any(arrays['external_applied']) and not np.any(arrays['qfrc_applied'])
            expected_apply=np.concatenate([np.zeros((1,90)),np.asarray([r['command'] for r in decoded])[:-1]])
            assert np.array_equal(np.asarray([r['command'] for r in applications]),expected_apply) and np.array_equal(np.asarray(ctrls),np.repeat(expected_apply,10,axis=0))
            assert all(r['applied_start_s']+1e-12>=r['valid_from_s'] for r in applications);dump(out/'CALL_LEDGER.json',calls)
        run('original_mask')
        gate=compare_archive(out/'original_mask',REFERENCE/'decoder_original');dump(out/'GATE_ORIGINAL_MASK_EXACT.json',gate)
        if not gate['passed']:
            dump(out/'STOP.json',dict(reason='original_mask exact reproduction failed; candidate NOT executed',candidate_executed=False,calls=calls));return 2
        run('recalibrated_55_62')
        assert {k:v for k,v in initial_checks[0].items() if k!='arm'}=={k:v for k,v in initial_checks[1].items() if k!='arm'}
        dump(out/'INITIAL_INVARIANTS.json',dict(passed=True,arms=initial_checks,initial_carry=[0.]*90,neural_parameters_identical=True,background_identical=True,model_identical=True,sensory_gain_identical=1.,causal_schedule_identical=True,decoder='same actual manifest/filter5ms/max1/denominator450; gains55,62 sole change',exact_historical_gate=True,runtime_mapping_check_passed=True))
        assert calls==BUDGET
        unchanged=all(sha(p)==h for p,h in hashes.items());assert unchanged,'source changed during run'
        dump(out/'COMPLETE.json',dict(capture_complete=True,gate_original_mask_exact=True,calls=calls,source_unchanged=unchanged,actual_step_argument_recorded=True,background_retained=True,auxiliary_force=False,extra_stimulation=False,training=False,non_neural_controller=False,biological_calibration='NOT_ESTABLISHED',behavior='NOT_ESTABLISHED',gain_only_changed_indices=[55,62],runtime_mapping_prerequisites_passed=True))
    except BaseException:
        dump(out/'CALL_LEDGER.json',calls);dump(out/'FAILED.json',dict(traceback=traceback.format_exc(),calls=calls));raise
    return 0


if __name__=='__main__':raise SystemExit(main())
