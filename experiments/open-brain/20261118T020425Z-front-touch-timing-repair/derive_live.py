"""Derive the closed-loop experiment from the archived live runner."""
from pathlib import Path

here = Path(__file__).resolve().parent
prior = here.parent / "20261118T002600Z-ascending-behavior-comparison"
source = (prior / "live.py").read_text()
source = source.replace("from neural_runtime import *", """from pathlib import Path
import sys

REPAIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPAIR.parent / '20261118T002600Z-ascending-behavior-comparison'))
from neural_runtime import *
from timing import CanonicalEvents, FrontTouchNeck""")
source = source.replace("from circuit import Circuit\nfrom fixed_neck import FixedCircuit\n", "")
source = source.replace("MODES=('native_full','reference_boundary','translated','no_up')",
                        "MODES=('repaired','fixed_canonical','fixed_raw','matched_reference_boundary')")
start = source.index("class FrozenEncoder")
end = source.index("def measure", start)
source = source[:start] + source[end:]
source = source.replace("    enc=FrozenEncoder();encoder_parity=qualify_encoder(enc)\n", "")
source = source.replace("source_idx,(1,2,3)", "source_idx,range(4)")
source = source.replace("FixedCircuit({'ports':neck_ports},batch=4)",
                        "FrontTouchNeck({'ports':neck_ports},an_roots,matched_lanes=(0,3),feedback_lanes=(0,),batch=4)")
start = source.index("    # Reference ANs are owned")
end = source.index("    bodies=", start)
source = source[:start] + source[end:]
source = source.replace("        full_an_spikes=((2,),bool),central_hits", "        central_hits")
start = source.index("    drive=np.zeros")
end = source.index("    context_indices=", start)
source = source[:start] + """    drive=np.zeros((4,len(neck.roots)),dtype=np.float32)
    edrive=np.zeros((4,len(neck_ports)),dtype=bool)
    cut=np.zeros(len(neck_ports),dtype=bool)
    canonical=CanonicalEvents((4,2))
""" + source[end:]
source = source.replace("FIT/'BINDING.json',ENCODER_WEIGHT,BODY_WEIGHT", "FIT/'BINDING.json',BODY_WEIGHT")
source = source.replace("    pins=sources(extra_files);pins.update(assets)",
                        "    extra_files.update(REPAIR.glob('*.py'));extra_files.add(REPAIR/'PROTOCOL.zh-CN.md')\n    pins=sources(extra_files);pins.update(assets)")
source = source.replace("        encoder_parity=encoder_parity,source_indices", "        source_indices")
source = source.replace("""        boundary_feedback_excluded_lanes=[1,2,3],excluded_feedback_edges=int(weights[source_idx,:].nnz),
        native_full_brain_unchanged=True,teacher_body_encoder='new engineering BANC receptor->AN current; not archived native body interface'))""",
                        """        boundary_feedback_excluded_lanes=[0,1,2,3],excluded_feedback_edges=int(weights[source_idx,:].nnz),
        matched_source_lanes=[0,3],matched_feedback_lanes=[0],raw_arrival_lanes=[2]))""")
source = source.replace("        drive.fill(0.);ref_drive.fill(0.);aff_drive.fill(0.)", "        drive.fill(0.)")
start = source.index("        for c in range(2):\n            if use[1,c]")
end = source.index("        edrive.fill", start)
source = source[:start] + source[end:]
start = source.index("        filter_state+=")
end = source.index("        active=", start)
source = source[:start] + """        events=canonical.step(arrivals[:,an_cols])
        events[2]=arrivals[2,an_cols]
""" + source[end:]
source = source.replace("brain.step([active]*4,events,extra)", "brain.step([active]*4,events)")
source = source.replace("neural['rates'][tick]=rates", "neural['rates'][tick]=canonical.rate")
source = source.replace("        neural['full_an_spikes'][tick]=brain.spikes[0,source_idx].numpy().astype(bool)\n", "")
source = source.replace("nominal_reference_lif_ticks=n,native_receptor_lif_ticks=n", "matched_source_batch_ticks=n,matched_source_lane_ticks=2*n")
source = "\n".join(line for line in source.split("\n") if not line.startswith("ENCODER_WEIGHT="))
assert "reference_arrivals" not in source and "ref_state" not in source and "ENCODER_WEIGHT" not in source
(here / "live_repair.py").write_text(source)
