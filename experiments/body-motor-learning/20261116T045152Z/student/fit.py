"""Fit a bounded residual to qualified same-body teacher decisions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from residual import BASE, HERE, INPUTS, Residual, baseline_policy, base_sources, load, read, stand, wrap


def teacher_data(root, sources, archive_preflight):
    summary = root / "RESULT.json"
    decisions = root / "DECISIONS.npz"
    trace = root / "TRACE.npz"
    result = read(summary)
    if result.get("joint_sampled_pass") is not True or result["walk_ms"] < 9600:
        raise ValueError(f"Full physical teacher gate failed: {summary}")
    assert result["actual_ms"] == 1700 + result["walk_ms"]
    assert stand.sha(decisions) == result["decision_sha256"]
    assert stand.sha(trace) == result["trace_sha256"]
    paths = [summary, decisions, trace]
    if not archive_preflight:
        if result["start"] not in ("nominal", "lean_left", "moving_forward", "uneven_feet"):
            raise ValueError(f"Independent physical verification start in teacher data: {summary}")
        acceptance = root / "TEACHER_ACCEPTANCE.json"
        qualify = read(acceptance)
        if qualify["passed"] is not True:
            raise ValueError(f"Physical and FlyGym contribution certification failed: {acceptance}")
        for name, file in (("result", summary), ("trace", trace), ("decision", decisions),
                           ("native_audit", root / "NATIVE_AUDIT.json"),
                           ("source_effect", root / "SOURCE_EFFECT.json"),
                           ("source_provenance", root / "SOURCE_PROVENANCE.json")):
            assert stand.sha(file) == qualify[name + "_sha256"]
            paths.append(file)
        provenance = read(root / "SOURCE_PROVENANCE.json")
        assert read(root / "NATIVE_AUDIT.json")["physical_pass"] is True
        assert read(root / "SOURCE_EFFECT.json")["passed"] is True
        flygym = Path(provenance["source_trace_dir"]) / "TRACE.npz"
        reference = Path(provenance["source_ref"])
        assert stand.sha(flygym) == qualify["flygym_trace_sha256"] == provenance["source_trace_sha256"]
        assert stand.sha(reference) == qualify["source_ref_sha256"] == provenance["source_ref_sha256"]
        paths.extend([acceptance, flygym, reference])
    for p in paths:
        sources[str(p)] = stand.sha(p)
    with np.load(decisions, allow_pickle=False) as z:
        x, a, intent, ms = (z[k] for k in ("x", "action", "intent", "ms"))
        assert x.shape == (len(ms), INPUTS) and a.shape == (len(ms), 42)
        assert np.isfinite(x).all() and np.isfinite(a).all()
        assert (abs(a) <= 1.).all()
        np.testing.assert_array_equal(ms, np.arange(len(ms)) * 5)
        selected = (500 <= ms) & (ms < 500 + result["walk_ms"])
        np.testing.assert_array_equal(intent.astype(bool), selected)
        return x[selected], a[selected]


def fit(teachers, validation, output, updates, bound, seed, archive_preflight):
    base = baseline_policy()
    output.mkdir()
    baseline = base_sources()
    sources = baseline.copy()
    sources.update({str(p): stand.sha(p) for p in (HERE / "fit.py", HERE / "residual.py", HERE / "PREREG.zh-CN.md")})
    samples, commands = zip(*(teacher_data(path, sources, archive_preflight) for path in teachers), strict=True)
    x, teacher = np.concatenate(samples), np.concatenate(commands)
    old = np.asarray([base(row, 1) for row in x])
    delta = teacher - old
    y = np.clip(delta, -bound, bound)

    anchor = BASE / "PROTO_nominal/DECISIONS.npz"
    baseline_result = read(anchor.parent / "RESULT.json")
    assert stand.sha(anchor) == baseline_result["decision_sha256"]
    sources[str(anchor)] = stand.sha(anchor)
    with np.load(anchor, allow_pickle=False) as z:
        ax = z["x"][z["intent"] == 1]
    normalizer = np.concatenate((x, ax))
    model = Residual(normalizer.mean(axis=0), normalizer.std(axis=0).clip(.01), bound)
    Adam = torch.optim.Adam(model.parameters(), lr=.001)
    xx, yy, aa = (torch.as_tensor(a, dtype=torch.float32) for a in (x, y, ax))
    generator = torch.Generator().manual_seed(seed ^ 977501)
    extra = []
    for path in validation:
        vx, vy = teacher_data(path, sources, archive_preflight)
        extra.append((path, vx, vy))
    stand.put(output / "INPUTS.json", dict(teachers=[str(p) for p in teachers], validation=[str(p) for p in validation],
        baseline=baseline, inputs=sources, bound=bound, updates=updates, seed=seed, archive_preflight=archive_preflight,
        anchor_samples=len(ax), teacher_samples=len(x), semantics="Qualified continuous walking residual at same observation"))
    with (output / "JOURNAL.jsonl").open("x") as log:
        for step in range(updates + 1):
            loss = (0., 0.)
            if step:
                i = torch.randint(len(xx), (96,), generator=generator)
                j = torch.randint(len(aa), (48,), generator=generator)
                supervised = F.smooth_l1_loss(model(xx[i]), yy[i], beta=.06)
                magnitude = model(aa[j]).square().mean()
                total = supervised + .20 * magnitude
                Adam.zero_grad(set_to_none=True)
                total.backward()
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError("Non-finite residual gradient")
                Adam.step()
                loss = float(supervised.detach()), float(magnitude.detach())
            path = output / f"step_{step:04d}.pt"
            with path.open("xb") as stream:
                torch.save(dict(model=model.state_dict(), optimizer=Adam.state_dict(), rng_state=generator.get_state(),
                                baseline_sources=baseline, step=step, archive_preflight=archive_preflight), stream)
            log.write(json.dumps(dict(step=step, teacher_loss=loss[0], anchor_mse=loss[1], sha256=stand.sha(path))) + "\n")
            log.flush()

    def metrics(inputs, actions):
        reloaded = load(path)
        one = np.asarray([reloaded(z, 1) for z in inputs])
        two = np.asarray([base(z, 1) for z in inputs])
        old_error = np.abs(two - actions)
        new_error = np.abs(one - actions)
        active = old_error > .001
        return dict(frame_count=len(inputs), corrected_mae=float(new_error.mean()),
                    original_mae=float(old_error.mean()), active_action_element_fraction=float(active.mean()),
                    active_corrected_mae=float(new_error[active].mean()) if np.any(active) else None,
                    active_original_mae=float(old_error[active].mean()) if np.any(active) else None,
                    mean_correction=float(np.abs(one - two).mean()),
                    max_correction=float(np.abs(one - two).max()))

    in_memory = wrap(model)
    fresh = load(path)
    for row in x[::37]:
        np.testing.assert_array_equal(in_memory(row, 1), fresh(row, 1))
    for name, old_sha in sources.items():
        assert stand.sha(Path(name)) == old_sha
    report = dict(checkpoint=str(path), checkpoint_sha256=stand.sha(path), training=metrics(x, teacher),
                  validation={str(name): metrics(vx, vy) for (name, vx, vy) in extra},
                  above_bound_fraction=float(np.mean(np.abs(delta) > bound)), updates=updates,
                  serialized_replay_bitwise=True, archive_preflight=archive_preflight)
    stand.put(output / "RESULT.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--teachers", type=Path, nargs="+", required=True)
    p.add_argument("--validation", type=Path, nargs="*", default=[])
    p.add_argument("--updates", type=int, default=240)
    p.add_argument("--bound", type=float, default=.15)
    p.add_argument("--seed", type=int, default=37513)
    p.add_argument("--archive-preflight", action="store_true", help="Use the frozen archive only to check software wiring")
    a = p.parse_args()
    if not 0 < a.bound <= .25 or a.updates < 1:
        p.error("Expected positive updates and 0 < bound <= .25")
    teachers = [p.resolve() for p in a.teachers]
    validation = [p.resolve() for p in a.validation]
    assert len(set(teachers + validation)) == len(teachers + validation)
    if a.archive_preflight:
        if teachers != [BASE / "PROTO_nominal"] or validation:
            p.error("Archive preflight accepts only the frozen nominal prototype archive")
    elif not all(HERE.parent / "teacher" in p.parents for p in teachers + validation):
        p.error("Training requires qualified same-body teacher runs from the assigned teacher directory")
    else:
        train_starts = {read(p / "RESULT.json")["start"] for p in teachers}
        validation_starts = {read(p / "RESULT.json")["start"] for p in validation}
        if train_starts & validation_starts:
            p.error("Validation starts must be held out as whole physical runs")
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    fit(teachers, validation, a.output.resolve(), a.updates, a.bound, a.seed, a.archive_preflight)
