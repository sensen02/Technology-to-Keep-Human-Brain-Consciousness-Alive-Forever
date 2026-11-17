"""Check the frozen evaluation inputs, paired causal contrasts and accepted scope."""
from pathlib import Path

import numpy as np

from bench_runtime import HERE, turn


def read_results(group, names):
    results = {}
    for name in names:
        where = HERE / group / name
        result = turn.read(where/"RESULT.json")
        for key, file in (("eyes","EYES.npz"),("trace","TRACE.npz"),("transport","TRANSPORT.json"),("receiver","RECEIVER.npz")):
            assert turn.stand.sha(where/file) == result[key+"_sha256"], (group,name,key)
        results[name] = result
    return results


def matched(group, side, reports):
    folder = HERE / group
    nominal, cut, restore = [reports[side+"_"+arm] for arm in ("target","cut","restore")]
    normal_end = nominal["first_delivery_action_ms"]
    assert normal_end is not None
    normal_path = folder / (side+"_target")
    cut_path = folder / (side+"_cut")
    with np.load(normal_path/"TRACE.npz") as a, np.load(cut_path/"TRACE.npz") as b:
        for key in ("qpos","qvel","target","com"):
            np.testing.assert_array_equal(a[key][:normal_end+1],b[key][:normal_end+1])
    with np.load(normal_path/"EYES.npz") as a, np.load(cut_path/"EYES.npz") as b:
        n = int(np.sum(a["labels"][:,0] < normal_end))
        np.testing.assert_array_equal(a["rgb"][:n],b["rgb"][:n])
    cut_journal = turn.read(cut_path/"TRANSPORT.json")
    assert not any(row["event"] in ("enqueue","release") for row in cut_journal)
    post = turn.read(folder/(side+"_restore")/"TRANSPORT.json")
    opened = [row["step"] for row in post if row["event"]=="restore"]
    assert len(opened)==1
    assert all(row["capture_step"]>=opened[0] for row in post if row["event"]=="release")
    assert cut["first_intent_ms"]==nominal["first_intent_ms"]
    return dict(cut_direction_selected=cut["first_direction"], target_before_cut_deg=nominal["final_body_delta_deg"],
                cut_body_deg=cut["final_body_delta_deg"], restored_body_deg=restore["final_body_delta_deg"],
                cut_actions_absent=cut["first_delivery_action_ms"] is None,
                prefix_bitwise_ms=normal_end, first_restored_delivery_ms=restore["first_delivery_action_ms"],
                recovered=restore["passed"], all_pass=nominal["passed"] and cut["passed"] and restore["passed"])


def statistics():
    pin = turn.read(HERE/"EVAL_INPUTS.json")
    for path, expected in pin["source_sha256"].items():
        assert turn.stand.sha(HERE/path)==expected
    graph = turn.read(HERE/"OFFICIAL_GRAPH.json")
    for source in graph["sources"]:
        assert turn.stand.sha(Path(source["path"]))==source["sha256"]
    planned = turn.read(HERE/"EVAL_CASES.json")
    noisy_names = [line.split()[0] for line in (HERE/"noisy_names.txt").read_text().splitlines()]
    baseline = read_results("EVAL",planned)
    noisy = read_results("NOISY",noisy_names)
    for group in (baseline,noisy):
        for result in group.values():
            assert result["graph_sha256"]==pin["source_sha256"]["OFFICIAL_GRAPH.json"]
            assert result["checkpoint_sha256"]==pin["source_sha256"]["VISION_CAL.pt"]
            assert result["codec_sha256"]==pin["source_sha256"]["CODEC.pt"]
    counts = {sign:sum(baseline[f"{sign}{i:02d}"]["passed"] for i in range(12)) for sign in ("L","R")}
    controls = {key:record["passed"] for key,record in baseline.items() if not key[1:].isdigit()}
    pairs = {key:matched("EVAL",key,baseline) for key in ("L","R")}
    dropouts = sum(result["receiver_dropped"] for result in noisy.values())
    assert dropouts>0
    restored = {key:noisy[key]["passed"] for key in noisy_names}
    agree = all(count>=11 for count in counts.values()) and all(controls.values()) and all(p["all_pass"] for p in pairs.values())
    result = dict(result="PASS" if agree else "FAIL", nominal=counts, control_results=controls,
                  cut_and_restore=pairs, noisy_results=restored, noisy_pass=all(restored.values()),
                  noisy_drops=dropouts,
                  failed_main=[key for key,record in baseline.items() if not record["passed"]],
                  runs=len(baseline)+len(noisy),
                  physical_native_steps=sum(x["body_native_steps"] for x in (*baseline.values(), *noisy.values())),
                  setup_native_steps=sum(x["setup_native_steps"] for x in (*baseline.values(), *noisy.values())),
                  native_checked_runs=[group+"/"+key for group,data in (("EVAL",baseline),("NOISY",noisy))
                                       for key,r in data.items() if "native_checks" in r],
                  scope="Frozen physical body, head-side vision and source-grounded bilateral DNa02 subset with engineering activity dynamics.")
    turn.stand.put(HERE/"PHASE_RESULT.json",result)
    print(result)


if __name__ == "__main__":
    statistics()
