"""Analytic constant-current threshold fixture; not full-network convergence."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from engine.neural_cond import ConductanceNetwork, ConductanceParams


def first_crossing(current_nA, dt_ms, method, duration_ms=50):
    net = ConductanceNetwork(1, ConductanceParams(), dt_ms, voltage_method=method)
    for _ in range(round(duration_ms/dt_ms)):
        if net.step(current_nA)[0]:
            return net.t_ms
    return None


def analyze(current_nA=.03):
    p = ConductanceParams()
    equilibrium = p.rest_mV + current_nA/p.leak_uS
    analytic = None if equilibrium <= p.threshold_mV else -p.capacitance_nF/p.leak_uS * math.log(
        (equilibrium-p.threshold_mV)/(equilibrium-p.rest_mV))
    rows = []
    for dt in (.5, .25, .125, .0625):
        for method in ('backward_euler', 'frozen_exact'):
            crossing = first_crossing(current_nA, dt, method)
            rows.append(dict(dt_ms=dt, method=method, first_spike_ms=crossing,
                             analytic_error_ms=None if analytic is None or crossing is None else crossing-analytic))
    return dict(current_nA=current_nA, analytic_first_crossing_ms=analytic, rows=rows,
                limits=['Single neuron constant current, zero initial synaptic conductance/adaptation.',
                        'Only first spike evaluated; no delay/feedback/network propagation.',
                        'frozen_exact is an existing diagnostic method, not a production model change.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze()
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
