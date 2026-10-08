"""Allowlisted prototype delivery: no credentials or full data archives."""
from pathlib import Path
import hashlib,json,shutil
SRC=Path(__file__).resolve().parent
DST=Path('/home/sensen/Desktop/cell_wound_prototype/brain_isolation')
FILES=['BRAIN_ISOLATION_PLAN.md','audit_brain_regions.py',
       'README_BRAIN_ISOLATION.zh-CN.md','verify_brain_isolation.py','inspect_delivery.py',
       'engine/neural_hist.py','engine/multirate_scenario.py',
       'run_multirate_scenario.py','run_multirate_selftest.py',
       'engine/da_protocol.py','run_da_protocol.py','run_da_protocol_selftest.py',
       'outputs/brain_isolation/ROUND4.md',
       'analyze_touch_ascending.py','make_touch_figure.py',
       'data/flywire/banc_connectome.npz','data/flywire/fafb_connectome.npz',
       'data/flywire/fafb_sensorimotor_groups.npz',
       'engine/electrode_damage.py','run_electrode_damage.py','run_electrode_damage_selftest.py',
       'engine/neck_cut_data.py','run_neck_cut_scenario.py','run_neck_cut_selftest.py',
       'audit_evidence_consistency.py',
       'engine/touch_damage_scenario.py','run_touch_damage_scenario.py','run_touch_damage_selftest.py',
       'outputs/brain_isolation/evidence_consistency_audit.json',
       'outputs/brain_isolation/touch_damage_demo.png','outputs/brain_isolation/touch_damage_report.json',
       'outputs/brain_isolation/ROUND7.md',
       'outputs/brain_isolation/touch_damage_traces.npz','outputs/brain_isolation/touch_damage_selftest.json',
       'outputs/brain_isolation/touch_damage_selftest_demo_path.json',
       'outputs/brain_isolation/TOUCH_PATHWAY_zh-CN.md',
       'outputs/brain_isolation/ROUND6.md',
       'outputs/brain_isolation/electrode_damage_demo.png','outputs/brain_isolation/electrode_damage_report.json',
       'outputs/brain_isolation/electrode_damage_field.npz','outputs/brain_isolation/electrode_damage_selftest.json',
       'outputs/brain_isolation/neck_cut_demo.png','outputs/brain_isolation/neck_cut_report.json',
       'outputs/brain_isolation/neck_cut_traces.npz','outputs/brain_isolation/neck_cut_selftest.json',
       'outputs/brain_isolation/touch_ascending_report.json',
       'outputs/brain_isolation/touch_ascending_arrays.npz',
       'outputs/brain_isolation/touch_ascending_demo.png',
       'outputs/brain_isolation/EXTERNAL_EVIDENCE_TOUCH_ELECTRODE.md',
       'engine/synapse_map.py','run_synapse_map.py','run_synapse_map_selftest.py',
       'run_da_real_comparison.py','run_da_real_comparison_selftest.py',
       'outputs/brain_isolation/ROUND5.md',
       'outputs/brain_isolation/da_figure5_digitised.json',
       'outputs/brain_isolation/synapse_map_report.json','outputs/brain_isolation/synapse_map_summary.json',
       'outputs/brain_isolation/synapse_map_README.md','outputs/brain_isolation/synapse_map_demo.png',
       'outputs/brain_isolation/synapse_map_sites.npz','outputs/brain_isolation/synapse_map_selftest.json',
       'outputs/brain_isolation/synapse_map_raw.npz','outputs/brain_isolation/synapse_map_stream_stats.json',
       'outputs/brain_isolation/synapse_map_stream.log',
       'outputs/brain_isolation/da_real_comparison.png','outputs/brain_isolation/da_real_comparison_report.json',
       'outputs/brain_isolation/da_real_comparison_traces.npz',
       'outputs/brain_isolation/da_real_comparison_selftest.json',
       'outputs/brain_isolation/EVIDENCE_LEDGER.md',
       'outputs/brain_isolation/REAL_DATA_DA_EVIDENCE.md',
       'outputs/brain_isolation/multirate_comparison.png','outputs/brain_isolation/multirate_report.json',
       'outputs/brain_isolation/multirate_traces.npz',
       'outputs/brain_isolation/da_protocol_comparison.png','outputs/brain_isolation/da_protocol_report.json',
       'outputs/brain_isolation/da_protocol_traces.npz',
       'outputs/brain_isolation/ROUND3.md',
       'outputs/brain_isolation/injury_coupling_demo.png','outputs/brain_isolation/injury_coupling_metrics.json',
       'outputs/brain_isolation/injury_coupling_traces.npz',
       'engine/receptors.py','engine/graded_vision.py','engine/injury_tissue.py',
       'run_graded_vision_selftest.py','run_banc_graded_vision.py',
       'run_injury_tissue_selftest.py','run_injury_coupling_demo.py','run_neural_recording_demo.py',
       'outputs/brain_isolation/verification_report.json',
       'outputs/brain_isolation/neural_recording_demo.png','outputs/brain_isolation/neural_recording_metrics.json',
       'outputs/brain_isolation/neural_recording_traces.npz',
       'outputs/brain_isolation/graded_vision_banc.json','outputs/brain_isolation/graded_vision_banc.png',
       'outputs/brain_isolation/graded_vision_banc.npz','outputs/brain_isolation/graded_vision_selftest.json',
       'engine/isolation_scenario.py','run_isolation_scenario.py','run_isolation_selftest.py',
       'run_modulation_uncertainty.py','run_banc_conductance.py','run_real_morphology_demo.py','run_real_subtree_demo.py',
       'outputs/brain_isolation/real_morphology_subtree_sample.swc',
       'outputs/brain_isolation/real_morphology_subtree_source_rejected.swc',
       'outputs/brain_isolation/real_morphology_subtree_metrics.json',
       'outputs/brain_isolation/real_morphology_subtree_traces.npz',
       'outputs/brain_isolation/real_morphology_subtree_demo.png',
       'outputs/brain_isolation/real_morphology_subtree_README.md',
       'outputs/brain_isolation/integrated_comparison.png','outputs/brain_isolation/integrated_report.json',
       'outputs/brain_isolation/integrated_intact.npz','outputs/brain_isolation/integrated_sham.npz',
       'outputs/brain_isolation/integrated_neck_cut.npz','outputs/brain_isolation/integrated_eye_loss.npz',
       'outputs/brain_isolation/integrated_neck_cut_supported.npz',
       'outputs/brain_isolation/modulation_uncertainty.png','outputs/brain_isolation/modulation_uncertainty.json',
       'outputs/brain_isolation/MODULATION_EVIDENCE.md','outputs/brain_isolation/ROUND2.md',
       'outputs/brain_isolation/banc_conductance_demo.png','outputs/brain_isolation/banc_conductance_metrics.json',
       'outputs/brain_isolation/real_morphology_demo.png','outputs/brain_isolation/real_morphology_metrics.json',
       'run_cond_selftest.py','run_cable_geometry_selftest.py',
       'run_electrode_selftest.py','run_local_electrode_demo.py',
       'run_local_tissue_selftest.py','run_local_chemical_demo.py',
       'run_hybrid_selftest.py','run_active_cut_demo.py',
       'outputs/brain_isolation/active_cut_demo.png',
       'outputs/brain_isolation/active_cut_metrics.json',
       'outputs/brain_isolation/active_cut_traces.npz',
       'engine/__init__.py','engine/units.py','engine/params.py',
       'engine/layers.py','engine/profile.py','engine/rules.py',
       'engine/neural.py','engine/checks.py','engine/record.py',
       'engine/cable.py','engine/neural_cond.py',
       'engine/electrode.py','engine/local_tissue.py',
       'engine/neural_active.py','engine/neural_hybrid.py',
       'outputs/brain_isolation/anatomy_audit.json',
       'outputs/brain_isolation/PROGRESS.md','outputs/brain_isolation/ROUND1.md',
       'outputs/brain_isolation/local_electrode_demo.png',
       'outputs/brain_isolation/local_electrode_metrics.json',
       'outputs/brain_isolation/local_electrode_traces.npz',
       'outputs/brain_isolation/local_chemical_demo.png',
       'outputs/brain_isolation/local_chemical_metrics.json',
       'outputs/brain_isolation/local_chemical_traces.npz']

def main():
    DST.mkdir(parents=True,exist_ok=True)
    manifest=[];missing=[]
    for rel in FILES:
        src=SRC/rel
        if not src.is_file(): missing.append(rel);continue
        dst=DST/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        if src.is_symlink(): raise RuntimeError('symlink in delivery list')
        shutil.copy2(src,dst)
        data=dst.read_bytes()
        manifest.append({'path':rel,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    report={'status':'incremental research prototype, NOT completed whole goal',
            'files':manifest,'missing_optional_files':missing,
            'credentials':'explicit allowlist excludes cookies/tokens; no dataset archives',
            'dependencies':['numpy','scipy','matplotlib','neuron (active backend)'],
            'python_used':str(SRC/'venv/bin/python')}
    (DST/'MANIFEST.json').write_text(json.dumps(report,indent=2))
    print('Delivered',len(manifest),'files to',DST)
    print('Optional files absent:',missing)

if __name__=='__main__':main()
