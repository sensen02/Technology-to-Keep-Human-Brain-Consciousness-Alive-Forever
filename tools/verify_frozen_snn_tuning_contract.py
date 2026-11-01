"""Archived numerical parity and bounded-parameter probes, not optimization."""
from pathlib import Path
import sys,json,hashlib
from dataclasses import asdict
import numpy as np
from scipy.sparse import load_npz
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import engine.neural_tuning as tuning
from engine.neural_cond import ConductanceParams
S=Path('/home/sensen/Desktop/engine修复验收/s5-detailed-handover-measurement/20261028T121000Z')
A=Path('/home/sensen/Desktop/engine修复验收/s4-neural-boundary-replay/20261101T185129Z')
O=Path('/home/sensen/Desktop/engine修复验收/s4-snn-tuning-contract/20261101T214247Z')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item(),allow_nan=False)+'\n')
def main():
    O.mkdir(parents=True,exist_ok=False)
    inputs=[Path(__file__),R/'engine/neural_tuning.py',R/'engine/neural_cond.py',A/'IDENTITY.json',A/'IDENTITIES.npz',A/'ACTUAL_We.npz',A/'ACTUAL_Wi.npz',A/'BASELINE_REPLAY.npz',S/'NEURAL.npz',S/'NEURAL_BOUNDARY.npz']
    hashes={str(p):sha(p) for p in inputs}
    dump(O/'PREREGISTRATION.json',dict(scope='S4 finite-parameter API contract and saved-boundary neural parity; no training/body/runtime deployment',source_sha256=hashes,steps='20ms=40ticks from inherited1100ms boundary, same saved current held2ticks per1ms window',baseline='Direct ConductanceNetwork and neutral tuning spawn must have bitwise state/spike equality and match saved baseline40ticks',probes=[dict(exc_gain=.9),dict(exc_gain=1.1),dict(inh_gain=.9),dict(inh_gain=1.1),dict(threshold_shift_mV=-1),dict(threshold_shift_mV=1)],invariants='Root order, sparse E/I support, positive conductance, intrinsic constants, full inherited state and timing fixed. Invalid parameters/root order/ring rejected.',limits='Engineering bounds, not physiological calibration. Instantaneous parameter interventions at saved state with fixed sensory replay, not learned weights or causal physical performance. No autograd/optimizer/objective/maze-policy in API.'))
    roots=np.load(A/'IDENTITIES.npz')['root_ids'];motor=np.load(A/'IDENTITIES.npz')['motor_indices'];params=ConductanceParams(**json.loads((A/'IDENTITY.json').read_text())['conductance_params'])
    We=load_npz(A/'ACTUAL_We.npz');Wi=load_npz(A/'ACTUAL_Wi.npz');boundary=np.load(S/'NEURAL_BOUNDARY.npz');current=np.load(S/'NEURAL.npz')['sensory_current_nA'][:20]
    archived=np.unpackbits(np.load(A/'BASELINE_REPLAY.npz')['spikes_packed'][:40],axis=1)[:,:12000].astype(bool)
    counts=dict(network_instances=0,neural_steps=0,tier_build=0,physical_model=0,body_step=0,optimizer=0,training=0)
    original=tuning.ConductanceNetwork
    def counted(*args,**kwargs):counts['network_instances']+=1;return original(*args,**kwargs)
    tuning.ConductanceNetwork=counted
    try:
        reference=counted(len(roots),params,dt_ms=.5,threshold_offset_mV=boundary['threshold_offset_mV']);reference.set_connectivity(We,Wi)
        for field in tuning.STATE_FIELDS:getattr(reference,field)[:]=boundary[field]
        reference.steps=2200;reference.t_ms=1100.;reference._ring_i=0
        bridge=tuning.FrozenConnectomeTuning(roots,We,Wi,params=params,dt_ms=.5,threshold_offset_mV=boundary['threshold_offset_mV'])
        state=bridge.snapshot(reference);neutral=bridge.spawn(snapshot=state)
        raster=[]
        for tick in range(40):
            a=reference.step(current[tick//2]);b=neutral.step(current[tick//2]);counts['neural_steps']+=2
            assert np.array_equal(a,b) and np.array_equal(a,archived[tick]);raster.append(b.copy())
            for field in tuning.STATE_FIELDS:assert np.array_equal(getattr(reference,field),getattr(neutral,field)),field
            assert reference.steps==neutral.steps and reference.t_ms==neutral.t_ms and reference._ring_i==neutral._ring_i
        raster=np.asarray(raster);rows=[];probe_arrays={}
        specs=[dict(exc_gain=.9),dict(exc_gain=1.1),dict(inh_gain=.9),dict(inh_gain=1.1),dict(threshold_shift_mV=-1),dict(threshold_shift_mV=1)]
        for i,values in enumerate(specs):
            p=tuning.TuningParameters(**values);net=bridge.spawn(p,snapshot=state);spikes=[]
            for tick in range(40):spikes.append(net.step(current[tick//2]).copy());counts['neural_steps']+=1
            spikes=np.asarray(spikes)
            for matrix,base,gain in [(net.We,We,p.exc_gain),(net.Wi,Wi,p.inh_gain)]:
                assert np.array_equal(matrix.indptr,base.indptr) and np.array_equal(matrix.indices,base.indices)
                assert np.array_equal(matrix.data,base.data*gain) and np.all(matrix.data>0)
            rows.append(dict(probe=i,parameters=asdict(p),total_spikes=int(spikes.sum()),motor202_spikes=int(spikes[:,motor].sum()),spike_bits_differ_from_neutral=int((spikes!=raster).sum()),final_voltage_max_difference=float(abs(net.v-neutral.v).max())))
            probe_arrays[f'probe{i}_spikes_packed']=np.packbits(spikes,axis=1);probe_arrays[f'probe{i}_final_v']=net.v.copy()
        rejected=[]
        tests=[('zero_exc',lambda:bridge.spawn(tuning.TuningParameters(exc_gain=0))),('negative_inh',lambda:bridge.spawn(tuning.TuningParameters(inh_gain=-1))),('gain_above_bound',lambda:bridge.spawn(tuning.TuningParameters(exc_gain=5))),('nan_gain',lambda:bridge.spawn(tuning.TuningParameters(exc_gain=float('nan')))),('threshold_above_bound',lambda:bridge.spawn(tuning.TuningParameters(threshold_shift_mV=6)))]
        shuffled=dict(state);shuffled['root_ids']=state['root_ids'][::-1].copy();tests.append(('root_order',lambda:bridge.spawn(snapshot=shuffled)))
        badring=dict(state);badring['ring_i']=1;tests.append(('ring_clock',lambda:bridge.spawn(snapshot=badring)))
        badW=We.copy();badW.data[0]=-abs(badW.data[0]);tests.append(('negative_base_weight',lambda:tuning.FrozenConnectomeTuning(roots,badW,Wi)))
        duplicate=roots.copy();duplicate[1]=duplicate[0];tests.append(('duplicate_roots',lambda:tuning.FrozenConnectomeTuning(duplicate,We,Wi)))
        for name,action in tests:
            try:action()
            except ValueError:rejected.append(name)
            else:raise AssertionError('Invalid probe accepted: '+name)
        np.savez_compressed(O/'PROBES.npz',neutral_spikes_packed=np.packbits(raster,axis=1),neutral_final_v=neutral.v,roots=roots,motor_indices=motor,**probe_arrays)
        dump(O/'RESULT.json',dict(contract=bridge.contract(),baseline_bitwise_parity=True,baseline_40ticks_equal_archived=True,probes=rows,rejected=rejected,counts=counts,source_unchanged=hashes=={str(p):sha(p) for p in inputs}))
        assert hashes=={str(p):sha(p) for p in inputs};print(json.dumps(dict(probes=rows,rejected=rejected,counts=counts),indent=2))
    finally:tuning.ConductanceNetwork=original
if __name__=='__main__':main()
