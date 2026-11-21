"""Isolate source dynamics and the rate-to-event conversion on paired inputs."""
import numpy as np
import torch

from timing import CanonicalEvents, FrontTouchNeck, DT
from neural_runtime import BIDIR, FIT, read, sha, write
from pathlib import Path

HERE = Path(__file__).resolve().parent


def event_summary(reference, actual, tolerance=10):
    result = []
    for r, a in zip(reference.T, actual.T, strict=True):
        r, a = np.flatnonzero(r), np.flatnonzero(a)
        i = j = matches = 0
        while i < len(r) and j < len(a):
            delta = a[j] - r[i]
            if abs(delta) <= tolerance:
                matches += 1
                i += 1
                j += 1
            elif delta > 0:
                i += 1
            else:
                j += 1
        result.append(dict(reference_count=len(r), actual_count=len(a),
                           first_actual_ms=float(a[0]*DT*1000) if len(a) else None,
                           first_reference_ms=float(r[0]*DT*1000) if len(r) else None,
                           one_ms_matches=matches, f1=float(2*matches/max(len(r)+len(a),1))))
    return result


def convert(arrivals, state=None):
    encoding = CanonicalEvents((2,))
    events = np.zeros_like(arrivals)
    output = np.zeros(arrivals.shape)
    phase = np.zeros(2)
    filters = np.zeros((2, 3))
    alpha = -np.expm1(-DT / np.array([.005, .020, .050]))
    if state is not None:
        mix = state["mix_logits"].softmax(-1).numpy()
        gain = state["log_gain"].clamp(-2, 2).exp().numpy()
    for tick, row in enumerate(arrivals):
        if state is None:
            events[tick] = encoding.step(row)
            output[tick] = encoding.rate
        else:
            filters += alpha * (row[:, None] / DT - filters)
            output[tick] = -1000 * np.expm1(-gain * (filters * mix).sum(-1) / 1000)
            phase += output[tick] * DT
            events[tick] = phase >= 1.
            phase -= events[tick]
    return events, output


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    binding = FIT / "BINDING.json"
    catalog_path = BIDIR / "CATALOG.json"
    weights = FIT / "FIT/update250.pt"
    state = torch.load(weights, weights_only=True, map_location="cpu")["state"]
    front = [record["banc_root"] for record in read(binding)["records"]]
    catalog = read(catalog_path)["ports"]
    ports = {root: catalog[root] for root in front}
    pins = {str(p): sha(p) for p in (binding, catalog_path, weights, Path(__file__),
                                    HERE / "timing.py", HERE / "fixed_transport.py")}
    rows = []
    for case in ("train_left", "train_right", "held_left", "held_right"):
        path = FIT / "DATA" / case / "PAIRED.npz"
        pins[str(path)] = sha(path)
        assert pins[str(path)] == read(path.parent / "RESULT.json")["trace_sha256"]
        with np.load(path, allow_pickle=False) as data:
            ref = data["native_reference_arrivals"]
            fixed = data["student_arrivals"]
            use = data["receptor_stimuli"]
            rate = data["teacher_rate_hz"]
        target, reproduced_rate = convert(ref)
        assert np.array_equal(reproduced_rate, rate)
        fixed_canonical, fixed_rate = convert(fixed)
        trained, trained_rate = convert(fixed, state)
        same_source_trained, _ = convert(ref, state)
        rows.append(dict(case=case, ticks=len(ref), source_arrivals=event_summary(ref, fixed),
                         fixed_canonical=event_summary(target, fixed_canonical),
                         old_translated=event_summary(target, trained),
                         reference_source_old_translator=event_summary(target, same_source_trained),
                         fixed_raw=event_summary(target, fixed),
                         repair_target_conversion=event_summary(target, convert(ref)[0]),
                         rate_nrmse_fixed_canonical=float(np.linalg.norm(rate-fixed_rate)/np.linalg.norm(rate)),
                         rate_nrmse_old_translated=float(np.linalg.norm(rate-trained_rate)/np.linalg.norm(rate))))

    # Isolate the real axon cut on new replays of one held input, with identical somata.
    models = [FrontTouchNeck({"ports": ports}, front, (0,), (0,), batch=1) for _ in range(2)]
    columns = models[0].port_columns
    cut_on, cut_off = 4500, 9000
    stim = [models[0].validate_inputs([r for r, _ in ports[root]["incoming_receptors"]], "body")
            for root in front]
    output = np.zeros((2, len(use), 2), dtype=bool)
    emitted = np.zeros_like(output)
    pulses = np.zeros_like(output)
    converters = [CanonicalEvents((2,)) for _ in models]
    drive = np.zeros((1, len(models[0].roots)), dtype=np.float32)
    for tick, active in enumerate(use):
        drive.fill(0)
        for c, receptors in enumerate(stim):
            if active[c]:
                drive[0, receptors] = 68.75
        for lane, model in enumerate(models):
            mask = np.array([bool(lane and cut_on <= tick < cut_off), False])
            e, a, _, _ = model.step(drive, mask)
            output[lane, tick] = a[0]
            emitted[lane, tick] = e[0]
            pulses[lane, tick] = converters[lane].step(a[0])
    assert np.array_equal(output[0], ref)
    assert np.array_equal(emitted[0], emitted[1])
    first_recovered = np.flatnonzero(output[1, cut_off:, 0]) + cut_off
    assert np.array_equal(output[:, :, 1][0], output[:, :, 1][1])
    assert np.array_equal(pulses[:, :, 1][0], pulses[:, :, 1][1])
    assert not output[1, cut_on:cut_off, 0].any()
    assert not pulses[1, cut_on + 1500:cut_off, 0].any()
    assert len(first_recovered) and first_recovered[0] >= cut_off + models[0].neck_delay[0]
    # Drive the two identified returned-DN branches and verify their conductance
    # is received by the calibrated AN on the same tick as the fixed receiver.
    feedback_roots = [root for root, p in catalog.items()
                      if p["direction"] == "DN" and any(target in front for target, _ in p["outgoing"])]
    active_roots = ["720575941500851362", "720575941535086424"]
    assert set(active_roots) <= set(feedback_roots)
    feedback_ports = {root: catalog[root] for root in [*front, *active_roots]}
    model = FrontTouchNeck({"ports": feedback_ports}, front, (0, 1), (0,), batch=2)
    mask = np.zeros(len(feedback_ports), dtype=bool)
    drive = np.zeros((2, len(model.roots)), dtype=np.float32)
    physical_dn = np.zeros((2, len(feedback_ports)), dtype=bool)
    indexes = [model.port_index[root] for root in active_roots]
    peak = 0.
    for tick in range(250):
        physical_dn.fill(False)
        if tick in (20, 40):
            physical_dn[:, indexes[(tick-20)//20]] = True
        model.step(drive, mask, eon_output=physical_dn)
        feedback = model.front_sources.feedback.last_g
        np.testing.assert_allclose(feedback[:, 0], model.last_conductance[:, 0, model.front_sources.neck_cells],
                                   rtol=0., atol=1e-15)
        assert not feedback[:, 1].any()
        peak = max(peak, float(feedback.max()))
    assert peak > 0.
    assert all(sha(path) == digest for path, digest in pins.items())
    write(HERE / "TIMING_DIAGNOSIS.json", dict(
        rows=rows, source_sha256=pins, cut_passed=True, source_normal_exact_to_reference=True,
        actual_neck_ticks=2*len(use)+250, feedback_peak_conductance_ns=peak,
        feedback_matches_fixed_conductance=True, feedback_ports=active_roots,
        released_arrivals_after_cut_tick=int(first_recovered[0])))
    np.savez_compressed(HERE / "CUT_TRACE.npz", arrivals=output, emitted=emitted, brain_events=pulses)
    print("paired timing, physical axon cut and returned-DN conductance all verified")


if __name__ == "__main__":
    main()
