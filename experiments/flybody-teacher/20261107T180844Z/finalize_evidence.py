"""Check completed-round evidence and write machine-readable handoff only."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent;OUT=Path('/home/sensen/Desktop/engine修复验收/flybody-teacher/20261107T180844Z')
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    mapping=load(OUT/'mapping/MANIFEST.json')
    for name,entry in mapping['files'].items():
        for root in [ROOT/'mapping',OUT/'mapping']:
            p=root/name;assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],str(p)
    episodes=[]
    for tag in ['measured_small_000','measured_longest_07718']:
        p=OUT/tag;r=load(p/'RESULT.json');v=load(p/'VERIFICATION.json');vid=load(p/'VIDEO_MANIFEST.json')
        for name,h in r['files'].items():assert sha(p/name)==h
        assert v['status']=='INDEPENDENT_MEASURED_REPLAY_PASS' and max(v['max_errors'].values())==0
        assert sha(p/'BODY.npz')==vid['body_sha256'] and sha(Path(vid['video']))==vid['video_sha256']
        assert vid['render_com_max_error']==0 and vid['visual_review']!='pending'
        episodes.append(dict(tag=tag,controls=r['controls'],native_steps=r['native_steps'],duration_s=r['sim_s'],reference_end=r['reference_end'],nonclaw_peak_native=r['nonfoot_max'],root_error_max_cm=r['reference_root_error_max'],joint_RMS_rad=r['joint_tracking_RMSE_rad'],verification=v['status'],video=vid['video']))
    status=dict(status='ROUND6_MEASURED_TEACHER_AND_STRUCTURAL_AUDIT_VERIFIED_NOT_PLATFORM_COMPLETE',episodes=episodes,
        new_recorded_and_replayed_native_steps=sum(x['native_steps'] for x in episodes)*2,new_student_training_updates=0,
        dataset_small_duplicate_count=load(OUT/'PARTITION_AUDIT.json')['small_exact_matches'],independent_recording_split='NOT_ESTABLISHED_MISSING_IDS',
        mapping_files_checked_in_both_locations=len(mapping['files']),mapping_zero_verified_scalar_maps=True,
        parent_independent_fk_verifier_pass=True,goal='active',next='Implement constrained geometry-aware motion retargeting, then current-body physical validation; no direct59to42action transfer',
        cleaned_redundant_archive_bytes=3019999672,source_hashes={str(p):sha(p) for p in ROOT.glob('*.py')})
    with (OUT/'ROUND6_STATUS.json').open('x') as f:json.dump(status,f,indent=2)
    print(json.dumps(status,indent=2))

if __name__=='__main__':main()
