"""Saved-only verification; zero network/physics simulation."""
from pathlib import Path
import json,numpy as np
from engine.embodied.motor_output import MotorOutput
from run_command1ms_single import O,HISTORY,ROOT
from run_one_second_motor_pair import sha,dump

def main():
 p=O/'normal';body=np.load(p/'BODY.npz');iv=np.load(p/'INTERVALS.npz');ticks=np.load(p/'TICKS.npz');mapping=json.loads((O/'MAPPING.json').read_text());pr=json.loads((O/'PREREGISTRATION.json').read_text());checks={}
 checks['all_preregistered_sources_unchanged']=all(sha(Path(k))==v for k,v in pr['source_hashes'].items())
 checks['history_inputs_unchanged']=all(sha(Path(k))==v for k,v in pr['history_hashes'].items())
 np.testing.assert_array_equal(ticks['spikes'].reshape(250,2,-1).sum(1),iv['spikes']);checks['every_network_event_exactly_once']=True
 np.testing.assert_array_equal(body['command'][1:],np.repeat(iv['commands'],80,axis=0));checks['actual_body_command_matches_all_250_writes']=True
 np.testing.assert_allclose(iv['sensory_sample_start_s'],np.arange(50)*.005,rtol=0,atol=1e-14);np.testing.assert_array_equal(iv['motor_sensory_index'],np.repeat(np.arange(50),5));checks['sensory_refresh50_and_5ms_hold_schedule']=True
 np.testing.assert_array_equal(iv['commands'][1:],iv['motor_decoded_next_controls'][:-1]);assert not iv['commands'][0].any();np.testing.assert_array_equal(iv['motor_source_end_tick'],np.r_[-1,np.arange(2,500,2)]);checks['no_future_decoder_result_applied']=True
 decoded=[];d=MotorOutput(mapping)
 for k in range(0,500,2):
  ctrl,_=d.consume(ticks['spikes'][k:k+2],spike_dt_s=.0005,start_s=k*.0005,end_s=(k+2)*.0005);decoded.append(ctrl)
 np.testing.assert_array_equal(decoded,iv['motor_decoded_next_controls']);checks['production_decoder_all_readouts_exact']=True
 d=MotorOutput(mapping)
 for k in range(500):
  ctrl,_=d.consume(ticks['spikes'][k:k+1],spike_dt_s=.0005,start_s=k*.0005,end_s=(k+1)*.0005)
  np.testing.assert_array_equal(d.rates,ticks['rates_hz'][k]);np.testing.assert_array_equal(ctrl,ticks['decoded_controls'][k])
 checks['all_500_per_tick_rates_and_decoding_exact']=True
 old=np.load(HISTORY/'normal/BODY.npz');checks['initial_qpos_qvel_ctrl_act_exact']=all(np.array_equal(body[k][0],old[k][0]) for k in ['qpos','qvel','activation','command']);oldi=np.load(HISTORY/'normal/INTERVALS.npz');np.testing.assert_array_equal(iv['sensory_current_nA'][0],oldi['sensory_current_nA'][0]);checks['first_real_encoded_current_exact']=True
 np.testing.assert_allclose(body['ground_force_foot_world'].sum(1)+body['ground_force_nonfoot_world'],body['ground_force_world'],rtol=1e-12,atol=1e-12);checks['world_reactions_sum_exact_within1e12']=True
 checks['finite']=all(np.isfinite(body[k]).all() for k in body.files);checks['warnings_zero']=not body['warnings'].any();checks['activation_bounds']=bool(body['activation'].min()>=0 and body['activation'].max()<=1)
 historical=json.loads((Path('/home/sensen/Desktop/engine修复验收/s2-split-write/20261027T163000Z/DEFAULT_REGRESSION.json')).read_text());checks['historical_default5ms_regression_all18_pass']=all(historical.values())
 dump(O/'VERIFICATION.json',dict(checks=checks,passed=all(checks.values()),calls=dict(model=0,forward=0,step=0,neural=0,decoder_consume=750),historical_default_regression_path='/home/sensen/Desktop/engine修复验收/s2-split-write/20261027T163000Z/DEFAULT_REGRESSION.json',historical_default_regression_sha256=sha(Path('/home/sensen/Desktop/engine修复验收/s2-split-write/20261027T163000Z/DEFAULT_REGRESSION.json'))))
 assert all(checks.values());print('saved-only verification PASS',checks)
if __name__=='__main__':main()
