"""Audit the previously learned RF collision gate on retained block inputs."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from rollout import HERE, STUMBLE_SOURCE, stan, W
sys.path.insert(1, str(STUMBLE_SOURCE.parent/'student'))
from stumble_policy import load, local_features


def evaluate(output, runs):
    torch.set_num_threads(1)
    student = STUMBLE_SOURCE.parent/'student'
    reference = json.loads((student/'STUDENT_SUITE.json').read_text())
    checkpoint = Path(reference['checkpoint'])
    if stan.sha(checkpoint) != reference['checkpoint_sha256']:
        raise RuntimeError('Accepted earlier collision checkpoint changed')
    policy = load(checkpoint)
    result = {}
    for run in runs:
        metadata = json.loads((run/'RESULT.json').read_text())
        for name in ('TRACE.npz','DECISIONS.npz'):
            if stan.sha(run/name) != metadata['files'][name]:
                raise RuntimeError('Inputs differ from physics manifest')
        with np.load(run/'DECISIONS.npz',allow_pickle=False) as d:
            obs = local_features(d['x'],d['local'],policy.leg)
            with torch.inference_mode():
                logits, delta = policy.model(torch.as_tensor(obs,dtype=torch.float32))
            scores = logits.numpy()
            intent = d['intent'].astype(bool)
            condition = intent & (scores > policy.threshold)
            if policy.force_floor is not None:
                condition &= policy.force_sign*d['local'][:,policy.force_index] >= policy.force_floor
            ms = d['ms'].copy()
            with np.load(run/'TRACE.npz',allow_pickle=False) as trace:
                b = trace['block_native'][ms]
                i = 3  # Body leg order, RF.
                face = (b[:,6+i]+b[:,12+i]) >= .002*W
                top = b[:,i] >= .02*W
            regions = np.flatnonzero(condition)
            result[str(run)] = dict(positive_gate_decisions=int(condition.sum()),
                  first_gate_ms=int(ms[regions[0]]) if len(regions) else None,
                  true_side_load_decisions=int(face.sum()),
                  side_coincidence_decisions=int(np.count_nonzero(face & condition)),
                  top_coincidence_decisions=int(np.count_nonzero(top & condition)),
                  peak_logits_during_intent=float(scores[intent].max()),
                  max_RF_correction_during_positive=float(delta[condition].abs().max()) if len(regions) else 0.)
    stan.put(output,dict(kind='Offline re-use screening only. No block policy execution or qualification.',
             checkpoint=str(checkpoint),checkpoint_sha256=stan.sha(checkpoint), results=result))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    p.add_argument('runs',type=Path,nargs='+')
    a=p.parse_args()
    evaluate(a.output.resolve(),[r.resolve() for r in a.runs])
