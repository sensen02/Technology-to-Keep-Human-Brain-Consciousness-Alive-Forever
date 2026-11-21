"""Derive the live runner without modifying the frozen P1 files."""
from pathlib import Path
import argparse

here=Path(__file__).resolve().parent


def main(archive):
    assay=archive/"experiments/open-brain/20261118T031132Z-flat-neck-assay"
    source=(assay/"run_assay.py").read_text()
    source=source.replace("REPAIR = Path(__file__).resolve().parent",
                          "TRIAL = Path(__file__).resolve().parent\nREPAIR = Path(__import__('os').environ['P1_ARCHIVE_ROOT'])/'experiments/open-brain/20261118T031132Z-flat-neck-assay'")
    source=source.replace("from assay_timing import CanonicalEvents, FrontTouchNeck",
                          """sys.path.insert(0,str(REPAIR))
from assay_timing import CanonicalEvents, FrontTouchNeck
from treatment import ARMS, SEEDS, FIT as OBSERVER_FIT, AssayTreatment""")
    source=source.replace("from schedule import ARMS, CUT_ROOT, CUT_ON, RESTORE, SEEDS, AssaySchedule\n","")
    source=source.replace("    assert an_roots[0]==CUT_ROOT and bound['records'][0]['body_input_side']=='left'\n"
                          "    schedule=AssaySchedule(len(neck_ports),neck.port_index[CUT_ROOT])",
                          """    assert an_roots[0]=='720575941633589179'
    cut=np.zeros(len(neck_ports),dtype=bool)
    therapy=AssayTreatment(n)""")
    source=source.replace("target_blocked=((4,),bool),\n        scheduled_event=((4,),np.int8)",
                          """left_raw_hz=((4,),float),left_used_hz=((4,),float),
        estimated_gain=((4,),float),fused_left_prediction=((4,),float),true_clean_events=((4,2),bool)""")
    source=source.replace("    pins=sources(extra_files);pins.update(assets)",
                          """    extra_files.update({OBSERVER_FIT,TRIAL/'PROTOCOL.zh-CN.md',*TRIAL.glob('*.py')})
    pins=sources(extra_files);pins.update(assets)""")
    source=source.replace("target_root=CUT_ROOT,cut_on_tick=CUT_ON,restore_tick=RESTORE",
                          "target_root=an_roots[0],observer_fit_sha256=sha(OBSERVER_FIT),active_arms=[1,3]")
    source=source.replace("        cut=schedule.step(tick)\n        neural['target_blocked'][tick]=cut[:,schedule.target]\n"
                          "        neural['scheduled_event'][tick]=schedule.scheduled_event(tick)\n","")
    source=source.replace("        events=canonical.step(arrivals[:,an_cols])",
                          """        events=canonical.step(arrivals[:,an_cols])
        neural['true_clean_events'][tick]=events
        raw, used, gain, prediction, candidate = therapy.step(tick,canonical.rate[:,0])
        neural['left_raw_hz'][tick]=raw;neural['left_used_hz'][tick]=used
        neural['estimated_gain'][tick]=gain;neural['fused_left_prediction'][tick]=prediction
        assert bool(events[0,0])==bool(candidate[0])
        events[:,0]=candidate""")
    source=source.replace("        brain.step([active]*4,events)",
                          """        brain.step([active]*4,events)
        therapy.cue.update(brain.spikes[:,dn_idx].numpy().astype(bool),arrivals[:,an_cols[1]])""")
    assert "schedule." not in source and "therapy.step" in source
    (here/"run_live.py").write_text(source)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("archive",type=Path);main(p.parse_args().archive)
