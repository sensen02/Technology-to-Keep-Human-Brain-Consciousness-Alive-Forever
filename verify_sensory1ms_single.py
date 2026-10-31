"""Saved-only causality/filter/state verification; no simulator."""
from pathlib import Path
import json,numpy as np
from engine.embodied.motor_output import MotorOutput
from run_one_second_motor_pair import sha,dump
O=Path('/home/sensen/Desktop/engine修复验收/s4-sensory1ms-command5ms/20261027T225000Z');H=Path('/home/sensen/Desktop/engine修复验收/s2-release-250ms/20261027T161000Z/normal')
def main():
 p=O/'normal';b=np.load(p/'BODY.npz');iv=np.load(p/'INTERVALS.npz');t=np.load(p/'TICKS.npz');s=np.load(p/'SENSORY.npz');mp=json.loads((O/'MAPPING.json').read_text());pr=json.loads((O/'PREREGISTRATION.json').read_text());checks={}
 checks['all_preregistered_sources_unchanged']=all(sha(Path(k))==v for k,v in pr['source_hashes'].items())
 np.testing.assert_array_equal(t['spikes'].reshape(50,10,-1).sum(1),iv['spikes']);checks['all500neural_ticks_once']=True
 np.testing.assert_array_equal(b['command'][1:],np.repeat(iv['commands'],400,axis=0));np.testing.assert_array_equal(iv['commands'][1:],iv['decoded_next_controls'][:-1]);assert not iv['commands'][0].any();checks['all50effective_body_holds_previouscompleted_decode']=True
 np.testing.assert_array_equal(iv['applied_source_interval'],np.arange(-1,49));checks['no_future_commands']=True
 np.testing.assert_allclose(s['physical_time'],np.arange(250)*.001,rtol=0,atol=1e-12);np.testing.assert_array_equal(s['qpos'],b['qpos'][::80][:-1]);np.testing.assert_array_equal(s['qvel'],b['qvel'][::80][:-1]);checks['all250samples_completed_actual_body']=True
 np.testing.assert_array_equal(t['current_nA'],np.repeat(s['current_nA'],2,axis=0));np.testing.assert_array_equal(iv['sensory_current_nA'],s['current_nA'][::5]);checks['input_replaced_at1ms_held2ticks']=True
 d=MotorOutput(mp);decoded=[]
 for k in range(0,500,10):
  ctrl,_=d.consume(t['spikes'][k:k+10],spike_dt_s=.0005,start_s=k*.0005,end_s=(k+10)*.0005);decoded.append(ctrl)
 np.testing.assert_array_equal(decoded,iv['decoded_next_controls']);checks['production_all50decode_readouts_exact']=True
 d=MotorOutput(mp)
 for k in range(500):
  ctrl,_=d.consume(t['spikes'][k:k+1],spike_dt_s=.0005,start_s=k*.0005,end_s=(k+1)*.0005);np.testing.assert_array_equal(d.rates,t['rates_hz'][k]);np.testing.assert_array_equal(ctrl,t['decoded_controls'][k])
 checks['all500persistent_filterstates_exact']=True
 old=np.load(H/'BODY.npz');oi=np.load(H/'INTERVALS.npz');checks['initial_qpos_qvel_act_ctrl_exact']=all(np.array_equal(b[k][0],old[k][0]) for k in ('qpos','qvel','activation','command'));np.testing.assert_array_equal(s['current_nA'][0],oi['sensory_current_nA'][0]);checks['first_current_exact']=True
 np.testing.assert_allclose(b['ground_force_foot_world'].sum(1)+b['ground_force_nonfoot_world'],b['ground_force_world'],rtol=1e-12,atol=1e-12);checks['worldforce_partition']=True
 checks['all_body_finite']=all(np.isfinite(b[k]).all() for k in b.files);checks['no_warnings']=not b['warnings'].any();checks['activation_bounds']=bool(b['activation'].min()>=0 and b['activation'].max()<=1)
 dump(O/'VERIFICATION.json',dict(checks=checks,passed=all(checks.values()),exit_code=0 if all(checks.values()) else 1,decoder_consume=550,model=0,forward=0,step=0,neural=0));assert all(checks.values());print('saved-only sensory verification PASS',len(checks))
if __name__=='__main__':main()
