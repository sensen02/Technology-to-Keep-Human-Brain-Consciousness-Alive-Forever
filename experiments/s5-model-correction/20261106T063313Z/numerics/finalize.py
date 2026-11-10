"""Post-run presentation/manifest audit; never changes frozen experiment sources."""
import sys
from pathlib import Path
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run as r

def main():
    r.checkhash()
    p=r.O/'NUMERICAL_REFINEMENT.png'
    with Image.open(p) as im:im.load();image=dict(size=im.size,mode=im.mode,decoded=True,visual_inspection=False,reason='read_image attempted but model does not declare image input')
    before=r.js(r.O/'MANIFEST_SHA256.json');mismatch=[p for p,h in before.items() if r.sha(p)!=h]
    assert mismatch==[str(r.O/'REPORT.zh-CN.md')],mismatch
    summary=r.js(r.O/'SUMMARY.json');cost=[]
    for folder in sorted((r.O/'CONFIGS').iterdir()):
        wi=r.js(folder/'WARM_IDENTITY.json');dt=wi['dt_s'];rs=[x for x in summary['results'] if x['dt_s']==dt]
        cost.append(dict(dt_s=dt,warm_steps=wi['neural_steps'],warm_wall_s=wi['wall_s'],releases=len(rs),release_wall_s=sum(x['wall_s'] for x in rs),max_RSS_MiB=max(x['peak_rss_MiB'] for x in rs)))
    audit=dict(passed=True,input_hashes_unchanged=True,initial_manifest_checked=len(before),initial_expected_report_presentation_change=mismatch,source_changes_after_freeze=False,postrun_helper_identity=dict(path=str(Path(__file__)),sha256=r.sha(Path(__file__))),image=image,cost=cost,report_only_changes='Added controlled replay6000-step accounting, fixed broken table, documented unavailable visual inspection; all numerical evidence unchanged',main_job='bash-58 completed exit0 collected',verify_job='bash-60 completed exit0 collected',report_exit=0,no_background_jobs_remaining=True,importance='重要辅助')
    r.dump(r.O/'FINAL_AUDIT.json',audit)
    files=[]
    for p in sorted(list(r.O.rglob('*'))+list(r.SRC.glob('*.py'))):
        if not p.is_file() or p.name in ['FILE_INVENTORY.json','MANIFEST_SHA256.json']:continue
        aux=p.suffix in ['.png','.md','.log'] or p.name in ['DIAGNOSTICS.json','PROGRESS.json','FINAL_AUDIT.json','finalize.py']
        files.append(dict(path=str(p),sha256=r.sha(p),bytes=p.stat().st_size,importance='重要辅助' if aux else '核心必需',purpose='presentation/progress/delivery audit' if aux else 'reproduction/raw evidence/verification/identity'))
    r.dump(r.O/'FILE_INVENTORY.json',dict(files=files,cleanup=r.js(r.O/'CLEANUP_LEDGER.json'),unconfirmed=[]))
    manifest={x['path']:x['sha256'] for x in files};manifest[str(r.O/'FILE_INVENTORY.json')]=r.sha(r.O/'FILE_INVENTORY.json')
    r.dump(r.O/'MANIFEST_SHA256.json',manifest)
    assert all(r.sha(p)==h for p,h in r.js(r.O/'MANIFEST_SHA256.json').items())
    print('FINAL AUDIT',len(files),'inventory files',len(manifest),'hashes 0 mismatches',image,'cost',cost,flush=True)
if __name__=='__main__':main()
