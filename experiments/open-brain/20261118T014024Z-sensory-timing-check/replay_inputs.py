"""Replay archived AN events through matching external-AN brain lanes."""
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BEHAVIOR = HERE.parent / "20261118T002600Z-ascending-behavior-comparison"
sys.path.insert(0, str(BEHAVIOR))

from neural_runtime import (BIDIR, DT, ContextBrain, FrozenCPUBrain, context_groups,
                            eon, load_connectome, read, sha, verify_upstream, write)


def first_difference(a, b):
    rows = np.flatnonzero((a != b).reshape(len(a), -1).any(axis=1))
    return int(rows[0]) if rows.size else None


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    fits = read(BEHAVIOR / "FINAL_BEHAVIOR.json")
    assert fits["replay_exact"]
    verify_upstream()
    index, _ = eon.get_hash_tables(eon.path_comp)
    model = FrozenCPUBrain(4, load_connectome(index))
    admitted_file = BIDIR / "EON/RESULT.json"
    admitted = read(admitted_file)["admitted"]
    chosen = read(BEHAVIOR / "PREFLIGHT/RESULT.json")["chosen"]
    context = np.asarray(context_groups(index)[chosen])
    # Exchange the prior reference/translated lane positions and duplicate both streams.
    source_lanes = [2, 2, 1, 1]
    ticks = 3600
    results = []
    pins = {str(path): sha(path) for path in (
        Path(__file__), BEHAVIOR / "neural_runtime.py", BEHAVIOR / "live.py",
        Path(eon.__file__), Path(eon.path_comp), Path(eon.path_con),
        admitted_file, BEHAVIOR / "FINAL_BEHAVIOR.json",
    )}
    for seed in (10901, 11311):
        folder = BEHAVIOR / f"FORMAL/seed{seed}"
        inputs = read(folder / "INPUTS.json")
        record = read(folder / "RESULT.json")
        neural_path = folder / "NEURAL.npz"
        assert sha(neural_path) == record["neural_sha256"]
        pins[str(neural_path)] = record["neural_sha256"]
        for key in ("INPUTS.json", "RESULT.json"):
            pins[str(folder / key)] = sha(folder / key)
        columns = [index[int(admitted[root])] for root in inputs["dn_roots"]]
        with np.load(neural_path, allow_pickle=False) as neural:
            hits = neural["central_hits"][:ticks]
            events = neural["brain_an_events"][:ticks]
            expected = neural["dn_emitted"][:ticks]
            rate = neural["rates"][:ticks]
            population = neural["population"][:ticks]
        random_context = np.random.default_rng(np.random.SeedSequence([seed, 18041]))
        expected_hits = random_context.random(hits.shape) < .01
        tick_index = np.arange(ticks)
        expected_hits[(tick_index < inputs["context_on_tick"]) |
                      (tick_index >= inputs["context_off_tick"])] = False
        assert np.array_equal(hits, expected_hits)

        brain = ContextBrain(model, inputs["source_indices"], range(4))
        emitted = np.zeros_like(expected)
        pop = np.zeros_like(population)
        torch_rng = torch.random.get_rng_state()
        for tick, selected in enumerate(hits):
            active = context[selected].tolist()
            brain.step([active] * 4, events[tick, source_lanes])
            emitted[tick] = brain.spikes[:, columns].numpy().astype(bool)
            pop[tick] = brain.spikes.sum(1).numpy()
        assert torch.equal(torch_rng, torch.random.get_rng_state())
        assert np.array_equal(emitted, expected[:, source_lanes])
        assert np.array_equal(pop, population[:, source_lanes])
        dn_tick = first_difference(expected[:, 1], expected[:, 2])
        an_tick = first_difference(events[:, 1], events[:, 2])
        assert an_tick < dn_tick
        differing_ports = np.flatnonzero(expected[dn_tick, 1] != expected[dn_tick, 2])
        physical = []
        for mode, lane in (("reference_boundary", 1), ("translated", 2)):
            path = folder / mode / "TRACE.npz"
            trace_hash = read(folder / mode / "RESULT.json")["trace_sha256"]
            assert sha(path) == trace_hash
            pins[str(path)] = trace_hash
            with np.load(path, allow_pickle=False) as body:
                physical.append({key: body[key][:ticks] for key in
                                 ("normal_load", "command", "qpos")})
        timing = {key: first_difference(physical[0][key], physical[1][key])
                  for key in physical[0]}
        assert all(dn_tick < tick for tick in timing.values())
        results.append(dict(seed=seed, first_AN_input_tick=an_tick,
                            first_AN_rate_difference_tick=first_difference(rate[:, 1], rate[:, 2]),
                            first_DN_difference_tick=dn_tick,
                            first_DN_difference_ports=[inputs["dn_roots"][i] for i in differing_ports],
                            first_physical_differences=timing,
                            same_PFL3_events=True, replayed_DN_and_population_bitexact=True,
                            torch_rng_unchanged=True))
        np.savez_compressed(HERE / f"seed{seed}_REPLAY.npz", dn_emitted=emitted,
                            population=pop, source_lanes=source_lanes)
        print(results[-1], flush=True)
    assert all(sha(path) == digest for path, digest in pins.items())
    write(HERE / "RESULT.json", dict(tick_seconds=DT, ticks_per_seed=ticks,
          brain_batch_ticks=ticks * len(results), brain_lane_ticks=ticks * len(results) * 4,
          neck_ticks=0, body_steps=0, optimizer_updates=0, rows=results, source_sha256=pins,
          replay_sha256={f"seed{row['seed']}_REPLAY.npz":
                         sha(HERE / f"seed{row['seed']}_REPLAY.npz") for row in results}))


if __name__ == "__main__":
    main()
