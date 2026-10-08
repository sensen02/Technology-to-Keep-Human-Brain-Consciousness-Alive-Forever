"""Figure for the body-touch ascending analysis. Reads the report JSON only."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "brain_isolation"


def main():
    r = json.loads((OUT / "touch_ascending_report.json").read_text())
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 8.5), layout="constrained")

    # (a) where body vs head touch first lands
    body = r["body_touch_output_profile"]["by_target_super_class"]
    head = r["head_touch_output_profile"]["by_target_super_class"]
    keys = ["ventral_nerve_cord_intrinsic", "ascending", "motor",
            "central_brain_intrinsic", "descending", "sensory"]
    y = np.arange(len(keys))
    ax[0, 0].barh(y - 0.2, [body.get(k, {}).get("fraction", 0) * 100 for k in keys],
                  height=0.4, label=f"BODY sensilla (n={r['source_partition']['body_route']})", color="tab:orange")
    ax[0, 0].barh(y + 0.2, [head.get(k, {}).get("fraction", 0) * 100 for k in keys],
                  height=0.4, label=f"HEAD sensilla (n={r['source_partition']['head_route']})", color="tab:blue")
    ax[0, 0].set_yticks(y, [k.replace("_", " ") for k in keys], fontsize=8)
    ax[0, 0].set(xlabel="% of outgoing synapses", title="(a) Where mechanosensory axons actually land")
    ax[0, 0].legend(fontsize=8)

    # (b) the relay bottleneck inside the nerve cord
    rb = r["relay_bottleneck"]
    parts = [("stays in VNC", rb["to_vnc_intrinsic"]["fraction"]),
             ("to motor neurons", rb["to_motor"]["fraction"]),
             ("to ascending neurons", rb["to_ascending_like"]["fraction"])]
    ax[0, 1].pie([p[1] for p in parts], labels=[f"{p[0]}\n{p[1]*100:.1f}%" for p in parts],
                 colors=["tab:gray", "tab:green", "tab:red"], autopct=None,
                 textprops={"fontsize": 9})
    ax[0, 1].set_title(f"(b) VNC local output split\n({rb['vnc_intrinsic_outgoing_synapses']:,.0f} synapses; "
                       "only the red part crosses the neck)")

    # (c) what the brain receives
    bb = r["brain_input_budget"]
    tot = bb["central_brain_intrinsic_incoming_synapses"]
    labels = ["from ascending\n(neck-crossing)", "head sensilla\ndirect", "body sensilla\ndirect", "everything else"]
    vals = [bb["from_ascending_like"]["fraction"] * 100, bb["from_sensory_head_direct"]["fraction"] * 100,
            bb["from_body_mech_direct"]["fraction"] * 100, 0.0]
    vals[3] = max(0.0, 100 - sum(vals[:3]))
    ax[1, 0].bar(range(4), vals, color=["tab:red", "tab:blue", "tab:orange", "tab:gray"])
    for i, v in enumerate(vals):
        ax[1, 0].text(i, v, f"{v:.2f}%" if v < 10 else f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    ax[1, 0].set_xticks(range(4), labels, fontsize=8)
    ax[1, 0].set(ylabel="% of central-brain-intrinsic input",
                 title=f"(c) Touch is a SMALL share of brain input\n(total input {tot:,.0f} synapses)")

    # (d) the neck cut as severed bandwidth
    nc = r["neck_cut"]
    ax[1, 1].bar(["crossing\nneurons", "edges\nsevered", "synapses\nsevered"],
                 [nc["crossing_neurons"], nc["edges_severed"], nc["synapses_severed"] / 1000],
                 color=["tab:purple", "tab:brown", "tab:red"])
    for i, (v, txt) in enumerate(zip(
            [nc["crossing_neurons"], nc["edges_severed"], nc["synapses_severed"] / 1000],
            [f"{nc['crossing_neurons']:,}", f"{nc['edges_severed']:,}", f"{nc['synapses_severed']/1e6:.2f} M"])):
        ax[1, 1].text(i, v, txt, ha="center", va="bottom", fontsize=9)
    ax[1, 1].set(ylabel="count (synapses in thousands)",
                 title="(d) What a neck cut actually severs\n"
                       f"{nc['crossing_neurons_ascending_like']:,} ascending + "
                       f"{nc['crossing_neurons_descending_like']:,} descending neurons")

    fig.suptitle(
        "BODY TOUCH -> BRAIN (real BANC labels, synapse-weighted, UNSIGNED edges)\n"
        "Figure-read from real annotations; the neck cut is an annotated-class edge removal, NOT a reconstructed cut plane.\n"
        "Reachability is deliberately NOT used as evidence: the cached graph is dense and polysynaptic, so connectivity alone cannot prove a pathway.",
        fontsize=9)
    fig.savefig(OUT / "touch_ascending_demo.png", dpi=150)
    print("wrote", OUT / "touch_ascending_demo.png")


if __name__ == "__main__":
    main()
