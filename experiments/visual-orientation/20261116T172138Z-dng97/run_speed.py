"""Run DNg97 source-qualified bilateral engineering velocity calibration."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from audit_source import HERE, NECK, PROJECT, ROOT, sha

SPEED = PROJECT/'experiments/body-motor-learning/20261116T075254Z/speed'
ARCHIVE = NECK/'20261112T225331Z/dnp09_interface'
sys.path[0:0] = [str(SPEED), str(ARCHIVE)]
from source_transport import SourceTransport
import pace_bench
from pace_lookup import ObservedPhase
from pace_bench import WARM, profile_value

GRAPH = HERE/'OFFICIAL_GRAPH.json'
BINDING = HERE/'DNg97_BINDING.json'
PROFILE = SPEED/'PACE_RAMP.json'
SCALE = .10  # Assigned velocity per unit common-mode activity, native length/s.
MIN_DELTA = .004


class VelocityReceiver:
    def __init__(self):
        content = json.loads(GRAPH.read_text())
        self.roots = [record[0] for record in ROOT.values()]
        assert self.roots == content['roots']
        row = {root:i for i,root in enumerate(self.roots)}
        col = {target['root_888']:i for i,target in enumerate(content['targets'])}
        counts = np.zeros((2,len(col)))
        for edge in content['connections']:
            counts[row[edge['pre']],col[edge['post']]] = edge['synapses']
        assert np.linalg.matrix_rank(counts)==2 and np.all(counts.max(0)>0)
        self.projection = counts/counts.max(0)
        # Exact-observation calibration. Biology supplied graph/identities,
        # while non-negative gains, normalization and readout are engineering.
        projected_common = np.ones(2)@self.projection
        self.readout = projected_common / (projected_common@projected_common)

    def velocity(self, packet):
        assert packet['root_ids'] == self.roots and packet['direction']=='DN'
        activity = packet['values']@self.projection
        return float(np.clip(activity@self.readout,0.,1.))*SCALE


def run(output, cut):
    torch.set_num_threads(1)
    config = json.loads(PROFILE.read_text())
    neck = SourceTransport(BINDING, sha(BINDING))
    receiver = VelocityReceiver()
    if cut:
        neck.set_cut('DN',True)
    deliveries = []
    decoder_records = []
    latest = 0.
    ticks = 0

    def send_advance():
        nonlocal latest
        speed = profile_value(config,neck.step-WARM)
        amplitude = float(speed/SCALE)
        assert 0 <= amplitude <= 1.
        if not cut:
            neck.send(dict(direction='DN',root_ids=neck.ports['DN'],values=[amplitude,amplitude],
                           capture_step=neck.step,source_bindings_sha256=neck.digest,
                           **{'from':'brain','to':'neck'}))
        for delivery in neck.advance():
            assert delivery['release_step']-delivery['capture_step']==2
            latest = receiver.velocity(delivery)
            deliveries.append((delivery['capture_step'],delivery['release_step'],latest))

    class Controller(ObservedPhase):
        def __call__(self, observation, requested):
            nonlocal ticks
            while neck.step < ticks:
                send_advance()
            expected = 0. if cut or ticks < 2 else profile_value(config,ticks-2-WARM)
            assert abs(expected-latest) < 1e-12
            decoder_records.append((ticks,float(requested[0]),latest))
            ticks += 5
            return super().__call__(observation,[latest,0.])

    original_capture = pace_bench.stan.identity.capture
    original_controller = pace_bench.PhaseFollower

    def capture(core, metadata, *, extra_paths):
        extra = [Path(__file__), GRAPH, BINDING, SPEED/'pace_lookup.py', SPEED/'pace_extended.py',
                 ARCHIVE/'source_transport.py', ARCHIVE/'api.py']
        return original_capture(core,{**metadata,'scope':'DNg97 delivered input engineering pacing', 'cut':cut},
                                extra_paths=[*extra_paths,*extra])

    pace_bench.PhaseFollower=Controller
    pace_bench.stan.identity.capture=capture
    try:
        pace_bench.roll(output,PROFILE)
    finally:
        pace_bench.PhaseFollower=original_controller
        pace_bench.stan.identity.capture=original_capture
    while neck.step<ticks:
        send_advance()
    link = neck.verify_final()
    physical = json.loads((output/'RESULT.json').read_text())
    delivered = np.asarray(deliveries,dtype=float).reshape(-1,3)
    records = np.asarray(decoder_records,dtype=float)
    np.savez_compressed(output/'DNG97_DELIVERY.npz', decision=records, delivery=delivered)
    assert ticks == physical['actual_ms'] and len(records)*5==ticks
    windows = physical['windows']
    low,high,again = [windows[label]['signed_forward_native_s']
                      for label in ('0-2400','4000-7200','8800-9600')]
    if cut:
        stationary = physical['initial_stand']['passed'] and physical['terminal_stand']['passed']
        passed = bool(stationary and abs(physical['walk']['signed_forward_native_s'])<.005
                      and not len(delivered) and np.all(records[:,2]==0.) and records[:,1].max()>0.)
        checks=dict(stationary=stationary,no_delivery=not len(delivered))
    else:
        checks=dict(native_course=physical['native_joint_pass'],accelerated=high-low>=MIN_DELTA,
                    decelerated=high-again>=MIN_DELTA,delay_verified=bool(np.all(delivered[:,1]-delivered[:,0]==2)),
                    one_stationary_start=True)
        passed=all(checks.values())
    report=dict(passed=passed,checks=checks,cut=cut,calibration='engineering common-mode bilateral',
                signed_speed_native_s=dict(low=low,high=high,again=again),
                speed_delta_gate=MIN_DELTA,bindings_sha256=sha(BINDING),graph_sha256=sha(GRAPH),
                link=link,delivered_packets=len(delivered),decision_count=len(records),
                source_physical_sha256=sha(output/'RESULT.json'),delivery_sha256=sha(output/'DNG97_DELIVERY.npz'),
                native_steps=physical['native_steps'],native_slip=physical['walk']['max_ground_normalized_slip'],
                native_physio_speed_mapping=False)
    (output/'DNG97_ACCEPTANCE.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--cut',action='store_true')
    cli = parser.parse_args()
    run(cli.output.resolve(),cli.cut)
