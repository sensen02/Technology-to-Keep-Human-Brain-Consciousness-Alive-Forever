"""Reuse the previous live runner and change only the brain-side event routing."""
from pathlib import Path

here=Path(__file__).resolve().parent
prior=here.parent/"20261118T042507Z-observer-closed-loop"
source=(prior/"run_live.py").read_text()
source=source.replace("from treatment import ARMS, SEEDS, FIT as OBSERVER_FIT, AssayTreatment",
"""sys.path.insert(0,str(TRIAL.parent/'20261118T042507Z-observer-closed-loop'))
from treatment import ARMS, SEEDS, FIT as OBSERVER_FIT, AssayTreatment
from anchored import AnchoredReadout""")
source=source.replace("true_clean_events=((4,2),bool)",
"""physical_encoded_events=((4,2),bool),advisory_left_hz=((4,),float),
        confirmed_readout_drift=((4,),bool),legacy_candidate_events=((4,),bool)""")
source=source.replace("canonical=CanonicalEvents((4,2))",
                      "anchor=AnchoredReadout(CanonicalEvents((4,2)),4,DT)")
source=source.replace("OBSERVER_FIT,TRIAL/'PROTOCOL.zh-CN.md',*TRIAL.glob('*.py')",
    "OBSERVER_FIT,TRIAL/'PROTOCOL.zh-CN.md',TRIAL.parent/'20261118T035424Z-adaptive-neck/adaptive_observer.py',*TRIAL.glob('*.py')")
source=source.replace("active_arms=[1,3]",
    """anchored_arms=[1,3],legacy_drift_control=2,
        biological_latency_reconstruction=False,advisory_detector_drives_encoder=False""")
start=source.index("        events=canonical.step(")
end=source.index("        active=context_indices", start)
source=source[:start]+"""        events=anchor.encode(arrivals[:,an_cols])
        neural['physical_encoded_events'][tick]=events
        raw, _, gain, prediction, candidate=therapy.step(tick,anchor.rates[:,0])
        # The physical axon trace and injected amplitude reading are separate
        # inputs to this simulation. No perturbation label enters the detector.
        advisory, confirmed=anchor.reconcile_left_amplitude(raw,np.ones(4,dtype=bool))
        neural['left_raw_hz'][tick]=raw
        neural['left_used_hz'][tick]=anchor.rates[:,0]
        neural['left_used_hz'][tick,2]=raw[2]
        neural['estimated_gain'][tick]=gain
        neural['fused_left_prediction'][tick]=prediction
        neural['advisory_left_hz'][tick]=advisory
        neural['confirmed_readout_drift'][tick]=confirmed
        neural['legacy_candidate_events'][tick]=candidate
        assert bool(events[0,0])==bool(candidate[0])
        events[2,0]=candidate[2]
""" + source[end:]
source=source.replace("=canonical.rate","=anchor.rates")
assert "events[:,0]=candidate" not in source and "canonical." not in source
(here/"run_live.py").write_text(source)
