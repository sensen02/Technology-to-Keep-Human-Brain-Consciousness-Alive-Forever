"""The one detailed-measurement trial requested: same protocol as 107000Z, 0.5 ms sampling.

Everything physical is byte-identical to 107000Z: 1.3 s of algorithmic walking with the
network live in the loop, algorithm withdrawn at 1.3 s, then 0.7 s neural-only with no PD,
no external force, no pose correction and no preset muscle command. The ONLY change is the
recording stride (0.5 ms instead of 1 ms) plus the extra measurements built from it.
"""
from pathlib import Path
import json, hashlib
import run_physical_walk_prewarm as w
P = Path(__file__).parent
O = Path('/home/sensen/Desktop/engine修复验收/s5-detailed-handover-measurement/20261028T121000Z')


def main():
    pre = O.parent / (O.name + '_PREDECLARED.json')
    pre.parent.mkdir(parents=True, exist_ok=True)
    assert not pre.exists() and not O.exists()
    sources = [Path(__file__), P / 'run_physical_walk_prewarm.py', P / 'engine/embodied/continuous_handover.py',
               P / 'engine/neural_cond.py', P / 'outputs/neural_rebuild/neural_rebuild_11_mn_pools.json',
               Path('/home/sensen/Desktop/engine修复验收/s4-integrated-repaired-closure/20261028T032000Z/INTEGRATED_MODEL.xml'),
               Path('/home/sensen/Desktop/engine修复验收/s5-neural-second-half/20261028T107000Z/BODY.npz')]
    pre.write_text(json.dumps(dict(
        scope='Single detailed measurement trial. Physics identical to 107000Z; only the recording stride changes to 0.5 ms.',
        same_protocol=dict(duration_s=1.3, warmup_settle_s=0., warmup_hold_s=0., neural_prewarm_s=1.1,
                           neural_only_s=.7, neural_command_dt_s=.001, keep_assistance_after_handover=False,
                           swing_height=.45, handover_s=1.3, sample_dt_s=.0005),
        model_provenance=dict(run_base='s4-integrated-repaired-closure/20261028T032000Z/INTEGRATED_MODEL.xml',
                              run_base_sha256='f15f92093c1a89cb201756f8141fc99b95dfea3c3cc6c63dea7479ce103890ab',
                              expected_built_model_sha256='3470c72f7e68f181ec1a80a5dc5fbff182b750091e7a1ddc7ccfbfb590b80cb6',
                              geometry_experiment_models_excluded=['s5-muscle-geometry-rebuild/20261028T113000Z/MODEL_REBUILT.xml',
                                                                   's5-contralateral-pathways/20261028T119000Z/MODEL_CONTRALATERAL.xml']),
        budget=dict(model_load=2, body_step=136000, neural_step=3000, network_instances=1, renders=1),
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}), indent=2) + '\n')
    w.run(O, 1.3, 100., 50., .0000125, neural_duration=.7, neural_command_dt_s=.001,
          warmup_settle_s=0., warmup_hold_s=0., neural_prewarm_s=1.1,
          keep_assistance_after_handover=False, swing_height=.45, sample_dt_s=.0005)


if __name__ == '__main__':
    main()
