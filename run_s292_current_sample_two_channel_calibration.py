"""S292/P3: explicit NEW current-sample calibration, not historical reproduction.

Import safe; dynamic work occurs only in main. Runtime mechanics are the exact
S291 implementation loaded from a registered local source, with explicit round,
arm-name and derivation-provider substitutions below. No historical file edited.
"""
from pathlib import Path
import hashlib
import json
import types
import numpy as np

ROOT=Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE=ROOT/'run_s291_two_channel_recalibration.py'
BASELINE=Path('/home/sensen/Desktop/engine修复验收/s289-decoder-range-support/20261026T173400Z/decoder_original')
TARGET_FILE=Path('/home/sensen/Desktop/engine修复验收/s244-chirality-of-the-5-of-6-arm/20261025T205824474993Z/CMD_A3_sym_matched.npz')
SCALE=1.236819199

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def calibrate(path,old,names):
    """One preregistered current sample only; no performance or row search."""
    neural=BASELINE/'NEURAL.npz';timing=BASELINE/'COMMAND_TIMING.npz'
    with np.load(neural,allow_pickle=False) as z:
        rates=z['rates_hz'].copy();raw=z['raw_rates_hz'].copy()
        tick_times=z['time_ms'].copy();tau=float(z['filter_tau_s']);den=float(z['decoder_rate_for_full_control_hz'])
        saved_gains=z['decoder_motor_gains'].copy()
    with np.load(timing,allow_pickle=False) as z:clock={k:z[k].copy() for k in z.files}
    with np.load(TARGET_FILE,allow_pickle=True) as z:
        targets=z['cmd'].copy();target_names=[str(x) for x in z['names']]
    assert target_names==names and targets.shape==(90,)
    assert rates.shape==(400,90) and raw.shape==(400,2,90)
    assert tau==.005 and den==450. and np.array_equal(saved_gains,old)
    expected={'source_start_s':.017,'source_end_s':.018,'valid_from_s':.018,'planned_applied_start_s':.018,'planned_applied_end_s':.019,'actual_applied_start_s':.018,'actual_applied_end_s':.019}
    for key,value in expected.items():assert abs(float(clock[key][17])-value)<1e-12,(key,clock[key][17])
    assert int(clock['applied_source_index'][18])==17 and int(clock['applied_source_interval'][18])==17
    assert abs(float(clock['applied_start_s'][18])-.018)<1e-12 and abs(float(clock['applied_end_s'][18])-.019)<1e-12
    assert np.allclose(tick_times[34:36],[17.5,18.],rtol=0,atol=1e-10)
    alpha=float(np.exp(-.0005/.005));filtered=np.zeros(90)
    for row in raw[:18]:
        for half in row:filtered=alpha*filtered+(1-alpha)*half
    assert np.allclose(filtered,rates[17],rtol=1e-13,atol=1e-12),'baseline own raw->filter recurrence mismatch'
    source_pools=json.loads((ROOT/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json').read_text())
    pools={r['actuator']:r for r in source_pools['measured']['resolved']}
    changed={55:'RMF_accesory_trochanter_flexor',62:'LHC_pleural_remotor_and_abductor'}
    entries=[];new=old.copy()
    for i,name in changed.items():
        assert names[i]==name and old[i]==0
        rate=float(rates[17,i]);q=rate/450.;original_target=float(targets[i]);target=original_target*SCALE
        assert np.isfinite(q) and q>0 and np.isfinite(target) and 0<target<1 and 0<original_target<1
        gain=target/q;assert np.isfinite(gain) and gain>0
        new[i]=gain
        entries.append(dict(index=i,actuator=name,gain_old=float(old[i]),gain_new=gain,q=q,filtered_rate_hz=rate,original_target=original_target,target=target,scale=SCALE,pool_root_ids=pools[name]['pool_root_ids'],raw_half_ms_hz=raw[17,:,i].tolist(),previous_filtered_rate_hz=float(rates[16,i]),recomputed_filtered_rate_hz=float(filtered[i]),formula='gain_new=(original_target*1.236819199)/(saved_current_rates_hz[17,index]/450)',denominator_stage='unclipped unit-gain preclip, NOT saved zero-masked command'))
    assert np.flatnonzero(new!=old).tolist()==[55,62] and np.count_nonzero(old==0)==26 and np.count_nonzero(new==0)==24
    sources=[Path(__file__).resolve(),BASE,neural,timing,TARGET_FILE,ROOT/'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json',ROOT/'engine/embodied/motor_output.py']
    hashes={str(p):sha(p) for p in sources}
    sample=dict(index=17,source_start_s=.017,source_end_s=.018,valid_from_s=.018,actual_applied_start_s=.018,actual_applied_end_s=.019,application_interval=18,applied_source_index=17,neural_sample_times_ms=tick_times[34:36].tolist(),filter_tau_s=.005,neural_dt_s=.0005,filter_alpha=alpha,filter_stage='after both own raw half-ms samples17[0] and17[1]',current_record=str(neural),timing_record=str(timing),historical_semantic_match_claim=False)
    calibration=dict(round='S292',passed=True,formula_status='EXPLICIT_NEW_CURRENT_SAMPLE_FORMULA_NOT_OLD_CALIBRATION_REPRODUCTION',candidate_gains_authorized=True,per_channel=entries,sample=sample,source_hashes=hashes,old_gains=old.tolist(),new_gains=new.tolist(),old_motor_gains=old.tolist(),new_motor_gains=new.tolist(),changed_indices=[55,62],original_zero_gain_count=26,candidate_zero_gain_count=24,full_control_hz=450.,scale=SCALE,target_source=str(TARGET_FILE),current_neural_source=str(neural),timing_source=str(timing),sample_search=False,denominator='saved S289 decoder_original rates_hz[17,i]/450 before gain and clipping',historical_sample_match='NOT_CLAIMED_NOT_REQUIRED_BY_NEW_FORMULA')
    return calibration,{r['index']:r for r in entries},new

def main(argv=None):
    # Keep the audited runtime byte-for-byte except the named experiment labels.
    # Compile does not execute main/model/tier work; this module never edits BASE.
    source=BASE.read_text(encoding='utf-8')
    assert source.count("run('recalibrated_55_62')")==1
    source=source.replace('S291','S292').replace('recalibrated_55_62','current_sample_55_62')
    runtime=types.ModuleType('s292_registered_runtime')
    runtime.__file__=str(Path(__file__).resolve())
    exec(compile(source,str(BASE),'exec'),runtime.__dict__)
    runtime.DERIVATION=BASELINE/'NEURAL.npz'
    runtime.read_derivation=calibrate
    original_dump=runtime.dump
    def dump(path,value):
        if path.name=='PREREG.json':
            value['calibration_kind']='explicit NEW current-sample unclipped denominator; no historical semantic match claim'
            value['runtime_base_source']=str(BASE);value['runtime_base_source_sha256']=sha(BASE)
            value['calibration_sample_index']=17;value['calibration_source']=str(BASELINE/'NEURAL.npz')
            value['parameter_diff']['only_change']='per_actuator_gain55/62:0 -> (S244target*1.236819199)/(S289current rates_hz[17]/450); other88 gains and all neural/physical/causal parameters unchanged'
            value['goal']='P3 explicit current-sample engineering interface calibration hypothesis, not restored biological connection or behavioral acceptance'
        if path.name=='DERIVATION.json':path=path.with_name('CALIBRATION.json')
        original_dump(path,value)
    runtime.dump=dump
    return runtime.main(argv)

if __name__=='__main__':raise SystemExit(main())
