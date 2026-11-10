"""Replay each completed pilot's neural/input records; no physics or learning."""
import argparse
import traceback
import numpy as np
from verify_saved import (HERE,BASE,DESKTOP,SOURCE,Verification,initialize,load,
                          read_json,write_json,model_checks,source_inputs,replay,snapshot)


def verify(v,mode):
    folder=DESKTOP/mode
    v.hashed(HERE/'PILOT_EXTENSION_PREREG.json')
    v.hashed(__file__)
    result,protocol=read_json(folder/'RESULT.json'),read_json(folder/'PROTOCOL.json')
    expected_condition='learned' if mode=='closed_loop' else 'hold'
    v.check('saved_condition',protocol['condition']==result['condition']==expected_condition)
    for name,expected in result['files'].items():v.hashed(folder/name,expected)
    v.hashed(folder/'RESULT.json')
    for path,key in ((BASE/'run_closed_loop.py','runner_sha256'),(DESKTOP/'readout/MODEL.npz','readout_sha256'),(BASE/'dynamics/module.py','network_sha256'),(BASE/'encoder/encoder.py','encoder_sha256')):v.hashed(path,protocol[key])
    policy,model=load(folder/'POLICY.npz'),load(DESKTOP/'readout/MODEL.npz')
    body,native,initial=load(folder/'BODY.npz'),load(folder/'NATIVE.npz')['rows'],load(folder/'INITIAL.npz')
    commands=load(folder/'COMMANDS.npz')['rows']
    enc,circuit=initialize(v)
    w=model_checks(v,model,enc,circuit)
    v.check('policy600',len(policy['time_s'])==600)
    indices=np.arange(600)*5
    inputs=source_inputs(body,native,initial,indices)
    v.equal('pilot_actual_source_q_velocity_load',inputs,policy['local_physical_input'])
    v.equal('pilot_start_time',body['time'][indices],policy['time_s'])
    v.equal('pilot_teacher_phase_flag',policy['teacher_warmup'],indices<1000)
    prediction=30*np.tanh((policy['mn_rates_hz']/100.)@w)
    v.equal('pilot_independent_predictions',prediction,policy['predicted_rate'],1e-10)
    source_commands=load(SOURCE/'COMMANDS.npz')['rows']
    source_body=load(SOURCE/'BODY.npz')
    v.hashed(SOURCE/'BODY.npz');v.hashed(SOURCE/'COMMANDS.npz')
    v.equal('pilot_exact_warmup_source_rates',commands[:1000,1:43],source_commands[:1000,1:43])
    warm_error=float(np.max(np.abs(body['qpos'][1000]-source_body['qpos'][1000])))
    v.equal('pilot_reported_warmup_qpos_error',warm_error,result['warmup_qpos_max_error'])
    if mode=='closed_loop':
        held=np.repeat(prediction[200:],5,axis=0)
        v.equal('pilot_posthandover_MN_only_commands',commands[1000:,1:43],held,1e-10)
    else:
        v.equal('pilot_posthandover_exact_zero_rate_hold',commands[1000:,1:43],np.zeros((2000,42)))
    v.check('pilot_current_shape_bounds',policy['sensory_current_nA'].shape==(600,556) and np.isfinite(policy['sensory_current_nA']).all() and np.all((policy['sensory_current_nA']>=0)&(policy['sensory_current_nA']<=.06)))
    vrange=replay(v,enc,circuit,inputs,policy['sensory_current_nA'],policy['mn_rates_hz'],policy['mn_voltages_mV'],load(folder/'HANDOVER.npz'))
    final=load(folder/'NEURAL_FINAL.npz')
    for key,value in snapshot(circuit).items():v.equal('full_neural_final_'+key,value,final[key])
    v.check('pilot_exact_neural_steps6000',v.steps==6000==result['neural_steps'])
    v.equal('pilot_voltage_range',vrange,result['neural_voltage_range_mV'])
    v.equal('pilot_MN_spikes',circuit.net.spike_count[circuit.mn_local].sum(),result['MN_spikes'])
    v.equal('pilot_active_MN',np.count_nonzero(circuit.net.spike_count[circuit.mn_local]),result['active_MN_count'])
    return v.summarize(samples=600,condition=expected_condition,voltage_range_mV=vrange,
                       voltage_minus90_sanity_pass=vrange[0]>=-90.,warmup_qpos_max_error=warm_error,
                       physical_replay='Not performed here; parent owns independent native physical replay.',
                       preserved_parent_physical_gate=result['physical_gate'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['closed_loop','output_hold'])
    mode=parser.parse_args().mode
    v=Verification(mode)
    output=HERE/('VERIFY_'+mode.upper()+'.json')
    if output.exists():raise FileExistsError(str(output))
    try:result=verify(v,mode)
    except Exception:
        failure=v.summarize(error=traceback.format_exc());failure['status']='VERIFICATION_FAILED'
        write_json(HERE/('FAILED_'+mode.upper()+'.json'),failure)
        raise
    write_json(output,result)
    print({k:result[k] for k in ('status','mode','checks_passed','actual_neural_replay_steps','voltage_range_mV')},flush=True)

if __name__=='__main__':main()
