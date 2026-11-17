"""Apply an externally specified propulsion pulse through the DNg100 source path."""
from pathlib import Path
import sys

import numpy as np
import torch

from forward_link import ForwardLink, BINDING, GRAPH, sha

HERE = Path(__file__).resolve().parent
BODY = HERE.parents[2] / "experiments/body-motor-learning/20261116T000536Z"
sys.path.insert(0, str(BODY))
from prototype import BANK, CHECKPOINT, policy
from walking_bench import rollout, stand

WALK_MS = 9600
STOP_MS = 1200


def run(output, condition):
    torch.set_num_threads(1)
    link = ForwardLink()
    stock = policy()
    selected = []
    if condition == "cut":
        link.switch(True)

    def motor(observation, intent):
        step = len(selected) * 5
        while link.transport.step < step:
            link.advance()
        link.send(intent)
        decoded = link.action()
        selected.append((step, intent, decoded))
        return stock(observation, decoded)

    result = rollout(output, "nominal", WALK_MS, motor)
    while link.transport.step < result["actual_ms"]:
        link.advance()
    journal = link.transport.journal()
    audit = link.transport.verify_final()
    received = [event for event in journal if event["event"] == "release"]
    first = next((step for step, _, active in selected if active), None)
    last = next((step for step, _, active in reversed(selected) if active), None)
    expected = result["actual_ms"] == 500 + WALK_MS + STOP_MS
    q = np.load(output / "TRACE.npz", allow_pickle=False)
    com = q["com"]
    displacement = float(np.linalg.norm(com[-1, :2] - com[0, :2]))
    motion = result["joint_sampled_pass"] and first == 505 and last == 10100
    still = not received and first is None and displacement < .01 and result["terminal_stand"]["passed"]
    passed = expected and (still if condition == "cut" else motion)
    evidence = dict(passed=bool(passed), scope="fixed source-bound propulsion pulse, body bench",
                    condition=condition, first_decoded_ms=first, last_decoded_ms=last,
                    measurements=len(received), displacement_native=displacement,
                    run=result, audit=audit,
                    sources={str(p):sha(p) for p in (Path(__file__), BINDING, GRAPH, BANK, CHECKPOINT)})
    stand.put(output / "PROBE.json", evidence)
    stand.put(output / "TRANSPORT.json", journal)
    np.save(output / "INTENTS.npy", selected)
    print(condition, passed, first, last, displacement, flush=True)
    return passed


if __name__ == "__main__":
    condition = sys.argv[1]
    assert condition in {"intact", "cut"}
    raise SystemExit(0 if run(HERE / ("PULSE_" + condition.upper()), condition) else 1)
