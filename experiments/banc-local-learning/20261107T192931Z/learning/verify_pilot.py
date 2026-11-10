"""Independent new-mechanism neural/history replay, no physical execution."""
from pathlib import Path
import hashlib
import json
import sys
import time
import traceback
import numpy as np
from runtime import Readout
from verify_cases import independent_features,Verify,load

HERE=Path(__file__).resolve().parent
ROUND=HERE.parent
OLD=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z')
DESKTOP=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T192931Z')
HOLD=Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z/output_hold')
SOURCE=Path('/home/sensen/Desktop/engine修复验收/flybody-retarget/20261107T182940Z/teacherderived_slow4_physics')
sys.path[:0]=[str(ROUND/'dynamics'),str(OLD/'encoder')]
from module import LocalCircuit
from encoder import SensoryEncoder


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out=HERE/'PILOT_VERIFICATION.json'
    if out.exists():raise FileExistsError(str(out))
    v=Verify();steps=0;started=time.monotonic()
    try:
        folder=DESKTOP/'closed_loop'
        protocol=json.loads((folder/'PROTOCOL.json').read_text());result=json.loads((folder/'RESULT.json').read_text())
        provenance={str(p):sha(p) for p in (Path(__file__),HERE/'verify_cases.py',HERE/'runtime.py',HERE/'PILOT_REPLAY_PREREG.json')}
        for name,expected in result['files'].items():
            p=folder/name;v.check('hash_'+name,sha(p)==expected);provenance[str(p)]=expected
        for p,key in ((HERE/'run_closed_loop.py','runner_sha256'),(HERE/'runtime.py','runtime_sha256'),(ROUND/'dynamics/module.py','network_sha256'),(OLD/'encoder/encoder.py','encoder_sha256'),(HERE/'INDEPENDENT_VERIFICATION.json','verification_sha256')):
            v.check('protocol_hash_'+str(p),sha(p)==protocol[key]);provenance[str(p)]=sha(p)
        model_path=Path(protocol['readout_path']);v.check('model_hash',sha(model_path)==protocol['readout_sha256'])
        provenance[str(model_path)]=sha(model_path)
        model=load(model_path);runtime=Readout(model_path);enc=SensoryEncoder();c=LocalCircuit()
        v.check('primary_newhistory',str(model['case'])=='newhistory' and bool(model['primary']))
        v.same('root_identity_encoder_circuit',enc.root_ids,c.root_ids)
        v.same('MN_root_identity',runtime.mn_root_ids,c.root_ids[c.mn_local]);v.same('MN_local_identity',runtime.mn_local,c.mn_local)
        v.same('MN_leg_identity',runtime.mn_leg,c.mn_leg)
        policy=load(folder/'POLICY.npz');body=load(folder/'BODY.npz');native=load(folder/'NATIVE.npz')['rows'];initial=load(folder/'INITIAL.npz');commands=load(folder/'COMMANDS.npz')['rows']
        v.check('600_samples',len(policy['time_s'])==600)
        k=np.arange(600)*5
        q=body['qpos'][k][:,initial['qa']];vel=body['qvel'][k][:,initial['va']];foot=np.zeros((600,6));foot[1:]=native[k[1:]*80-1,1:7]
        physical=np.concatenate((q,vel,foot),axis=1)
        v.same('source_local_input',physical,policy['local_physical_input']);v.same('source_time',body['time'][k],policy['time_s'])
        features=independent_features(policy['mn_rates_hz'],[0,1,4,10])
        v.same('independent_history_features',features,policy['readout_features'])
        predictions=30*np.tanh(features@model['weights'])
        v.same('independent_history_prediction',predictions,policy['predicted_rate'],1e-10)
        v.same('history_observation_count',policy['predictor_observations_seen'],np.arange(1,601))
        v.same('teacherphase',policy['teacher_warmup'],k<1000)
        source_cmd=load(SOURCE/'COMMANDS.npz')['rows'];v.same('warmup_applied_source',commands[:1000,1:43],source_cmd[:1000,1:43])
        v.same('learned_held_commands',commands[1000:,1:43],np.repeat(predictions[200:],5,axis=0),1e-10)
        source_body=load(SOURCE/'BODY.npz');warmerr=float(np.abs(body['qpos'][1000]-source_body['qpos'][1000]).max())
        v.same('warmup_source_qpos_error',warmerr,result['warmup_qpos_max_error'])
        # Historical control is eligible only as matched body/zero-command evidence.
        old_initial=load(HOLD/'INITIAL.npz');old_protocol=json.loads((HOLD/'PROTOCOL.json').read_text());old_commands=load(HOLD/'COMMANDS.npz')['rows']
        for key in initial:v.same('historical_hold_initial_'+key,initial[key],old_initial[key])
        for key in ('body_source_sha256','physics_dt','command_dt','policy_dt','hold_s','total_s'):
            v.check('historical_hold_protocol_'+key,protocol[key]==old_protocol[key])
        v.same('historical_hold_same_warmup_rates',commands[:1000,1:43],old_commands[:1000,1:43])
        v.same('historical_hold_postwarm_exactzero',old_commands[1000:,1:43],np.zeros((2000,42)))
        handover=load(folder/'HANDOVER.npz');zero=np.zeros(20000)
        maxima=dict(encoder_current=0.,MN_start_rate=0.,MN_start_voltage=0.,runtime_feature=0.,runtime_action=0.)
        vmin=0.;vmax=-1e10
        for tick in range(600):
            current=enc.encode(q[tick],vel[tick],foot[tick]);saved=policy['sensory_current_nA'][tick]
            maxima['encoder_current']=max(maxima['encoder_current'],float(np.abs(current[enc.local_indices]-saved).max()))
            maxima['MN_start_rate']=max(maxima['MN_start_rate'],float(np.abs(c.filtered_rates_hz[c.mn_local]-policy['mn_rates_hz'][tick]).max()))
            maxima['MN_start_voltage']=max(maxima['MN_start_voltage'],float(np.abs(c.net.v[c.mn_local]-policy['mn_voltages_mV'][tick]).max()))
            runtime.observe(c.filtered_rates_hz[c.mn_local])
            maxima['runtime_feature']=max(maxima['runtime_feature'],float(np.abs(runtime.feature_vector()-features[tick]).max()))
            maxima['runtime_action']=max(maxima['runtime_action'],float(np.abs(runtime.action()-policy['predicted_rate'][tick]).max()))
            if tick==200:
                for key,value in dict(neural_v=c.net.v,neural_ge=c.net.ge,neural_gi=c.net.gi,neural_g_adapt_uS=c.net.g_adapt,neural_delay_ring=c.net._ring,neural_ring_i=c.net._ring_i,mn_rates_hz=c.filtered_rates_hz[c.mn_local]).items():v.same('handover_'+key,value,handover[key])
                for key,value in runtime.state_dict().items():v.same('handover_predictor_'+key,value,handover['predictor_'+key])
            stimulus=np.zeros(20000);stimulus[enc.local_indices]=saved;c.advance(stimulus,zero,5.)
            steps=c.net.steps;vmin=min(vmin,float(c.net.v.min()));vmax=max(vmax,float(c.net.v.max()))
            if (tick+1)%200==0:print(json.dumps(dict(replayed_samples=tick+1,neural_steps=steps,wall_s=time.monotonic()-started)),flush=True)
        for key,error in maxima.items():v.check('trace_'+key,error<=(1e-10 if key=='runtime_action' else 0),max_abs_error=error)
        final=load(folder/'NEURAL_FINAL.npz')
        for key,value in dict(v=c.net.v,ge=c.net.ge,gi=c.net.gi,g_adapt_uS=c.net.g_adapt,refractory_until_ms=c.net.refractory_until_ms,spike_count=c.net.spike_count,delay_ring=c.net._ring,ring_i=c.net._ring_i,filtered_rates_hz=c.filtered_rates_hz,steps=c.net.steps).items():v.same('neural_final_'+key,value,final[key])
        predictor_final=load(folder/'READOUT_FINAL.npz')
        for key,value in runtime.state_dict().items():v.same('predictor_final_'+key,value,predictor_final[key])
        v.same('voltage_range',[vmin,vmax],result['neural_voltage_range_mV']);v.check('6000_steps',steps==6000==result['neural_steps'])
        v.same('MN_spikes',c.net.spike_count[c.mn_local].sum(),result['MN_spikes']);v.same('active_MN',np.count_nonzero(c.net.spike_count[c.mn_local]),result['active_MN_count'])
        v.check('no_physics_fit_runner_imports',not any(name in sys.modules for name in ('mujoco','fit_cases','run_closed_loop')))
        output=dict(status='PASS_NEW_NEURAL_HISTORY_REPLAY_NOT_WALKING',checks_passed=len(v.checks),checks=v.checks,trace_max_abs_errors=maxima,actual_neural_replay_steps=steps,new_physical_steps=0,new_fit_solves=0,voltage_range_mV=[vmin,vmax],voltage_minus90_sanity_pass=vmin>=-90,biological_calibration='NOT_ESTABLISHED',historical_hold_body_contract_matches=True,historical_hold_neural_comparison=False,warmup_qpos_max_error=warmerr,source_hashes=provenance,wall_s=time.monotonic()-started)
        out.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({k:output[k] for k in ('status','checks_passed','actual_neural_replay_steps','voltage_range_mV')}),flush=True)
    except Exception:
        (HERE/'FAILED_PILOT_VERIFICATION.json').write_text(json.dumps(dict(error=traceback.format_exc(),checks=v.checks,actual_neural_replay_steps=steps),indent=2)+'\n')
        raise

if __name__=='__main__':main()
