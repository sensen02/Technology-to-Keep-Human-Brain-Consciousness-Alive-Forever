"""Replay saved physical arrivals through the new P1 encoder without training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
FLAT = HERE.parent / "20261118T031132Z-flat-neck-assay"
FAULT = HERE.parent / "20261118T051014Z-neck-fault-assay"
BASELINE = HERE.parent / "20261117T051249Z-bidirectional" / "CATALOG.json"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("anchored_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.path[:0] = [str(args.anchored_dir), str(FLAT)]
    from anchored import AnchoredReadout
    from assay_timing import CanonicalEvents, DT

    pins = {str(path):sha(path) for path in
            (args.anchored_dir/"anchored.py", FLAT/"assay_timing.py", BASELINE)}
    ports = json.loads(BASELINE.read_text())["ports"]
    right_dn = "720575941500851362"
    checks, total_ticks, total_episodes = [], 0, 0
    for seed in (10901,11311,11701):
        trial = FAULT/"FORMAL"/f"seed{seed}"
        outcome = json.loads((trial/"RESULT.json").read_text())
        receipt = json.loads((trial/"INPUTS.json").read_text())
        archive = trial/"NEURAL.npz"
        assert sha(archive) == outcome["neural_sha256"]
        pins[str(archive)] = outcome["neural_sha256"]
        modes = receipt["modes"]
        with np.load(archive,allow_pickle=False) as data:
            actual = data["an_arrived"]
            reference, rate = data["brain_an_events"], data["rates"]
            on, off = receipt["fault_on_tick"], receipt["transient_restore_tick"]
            guard = on+int(round(.100/DT))
            confirmed = np.zeros(len(modes),dtype=int)
            for trusted in (True,False):
                readout = AnchoredReadout(CanonicalEvents(actual.shape[1:]),len(modes),DT)
                available = np.full(len(modes),trusted,dtype=bool)
                for tick, arrivals in enumerate(actual):
                    events = readout.encode(arrivals)
                    assert np.array_equal(events,reference[tick])
                    assert np.array_equal(readout.rates,rate[tick])
                    observed = rate[tick,:,0]*(.66 if on<=tick<off else 1.)
                    advisory, chosen = readout.reconcile_left_amplitude(observed,available)
                    expected = np.where(chosen,rate[tick,:,0],observed)
                    assert np.array_equal(advisory,expected)
                    if trusted:
                        assert chosen.any()==(not np.array_equal(advisory,observed))
                        confirmed += chosen
                    else:
                        assert not chosen.any()
            silence = slice(guard,off)
            for arm in ("tear_an","block_an"):
                index = modes.index(arm)
                assert not reference[silence,index,0].any()
            tear, transient = modes.index("tear_an"), modes.index("block_an")
            assert not actual[guard:,tear,0].any() and reference[off:,transient,0].any()
            # For an exact exponential accumulator, emitted mass through t is
            # cumulative arrivals minus the positive integrated residual after t.
            residual_events = []
            for tick in np.flatnonzero(reference[guard:,tear,0])+guard:
                assert rate[tick,tear,0]>0.
                assert reference[:tick+1,tear,0].sum()==actual[:tick+1,tear,0].sum()
                last = np.flatnonzero(actual[:tick+1,tear,0])[-1]
                residual_events.append(dict(tick=int(tick),last_raw_arrival_tick=int(last),
                                            rate_hz=float(rate[tick,tear,0])))
            upstream_modes = ("intact","tear_an","dropout_an","block_an")
            left_event_bits = {m:int(np.count_nonzero(reference[:,modes.index(m),0]!=reference[:,0,0]))
                               for m in upstream_modes}
            delivered = data["dn_arrived"][on+13:off,modes.index("block_dn")]
            direct = outcome["dn_roots"].index(right_dn)
            assert not delivered[:,direct].any()
            right_targets = set(dict(ports[right_dn]["outgoing"]))
            other_forward = {root:dict(group=ports[root]["group"],cell_type=ports[root]["cell_type"],
                                      side=ports[root]["side"],received=int(delivered[:,col].sum()),
                                      registered_targets_shared_with_right_dng100=len(
                                          right_targets.intersection(dict(ports[root]["outgoing"]))))
                for col,root in enumerate(outcome["dn_roots"])
                if root!=right_dn and ports[root]["group"] in {"forward","forward_speed"}}
            checks.append(dict(seed=seed, trusted_auxiliary_correction_ticks=dict(zip(modes,confirmed.tolist(),strict=True)),
                left_brain_event_bits_different_from_intact=left_event_bits,
                left_encoded_silence_after_initial_filter_tail_before_release_ticks=off-guard,
                late_phase_rounding_events_without_raw_arrivals=residual_events,
                surviving_forward_dn_arrivals_during_confirmed_right_block=other_forward))
            total_ticks += 2*len(actual)*len(modes)
            total_episodes += len(modes)
    result = dict(project="P1_open_source_brain", trace_arms=total_episodes,
        encoder_lane_steps=total_ticks, all_archived_event_bits_and_rates_exact=True,
        new_body_or_neck_or_brain_steps=0, optimizer_updates=0,
        counterfactual_aux_amplitude_multiplier=.66,
        verification_input_set_experimentally=True,
        interventions_replayed_not_retrained=True,
        input_sha256=pins, seeds=checks)
    with args.output.open("x",encoding="utf-8") as file:
        json.dump(result,file,ensure_ascii=False,indent=2)
        file.write("\n")
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
