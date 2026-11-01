"""Fetch the official extracted BANC individual behavior deposit, generation pinned.
Downloads data only; never executes supplied notebook/code or modifies a model.
"""
from pathlib import Path
import urllib.request,urllib.parse,json,hashlib,base64,time
from concurrent.futures import ThreadPoolExecutor
O=Path('/home/sensen/Desktop/engine修复验收/s4-banc-individual-behavior/20261101T211209Z')
BUCKET='lee-lab_brain-and-nerve-cord-fly-connectome'
LIST=f'https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?prefix=behavior%2F&maxResults=100'

def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def openurl(url):return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'BANC-data-reproducibility-audit/1.0'}),timeout=60)
def main():
    O.mkdir(parents=True,exist_ok=False)
    dump(O/'PREREGISTRATION.json',dict(task='S1/S3/S4 traceable same-individual behavioral data audit; no training or physical/neural simulation',official_paper='https://www.nature.com/articles/s41586-026-10735-w',official_description='https://raw.githubusercontent.com/htem/BANC-project/main/manuscript/print/dataverse/documentation/behavior.md',dataverse='https://doi.org/10.7910/DVN/7WTH1N',dataverse_file=13917712,acquisition='Official GCS extracted tree linked by the paper repository. Do not duplicate ZIP plus extracted raw files. Pin object generations and compare list MD5 and length.',list_url=LIST,checks='Every file byte count and base64 MD5 match official object metadata; retain all originals unmodified; no notebook execution',analysis_goal='Find the exact BANC individual selector and reproduce582 choices/~70%right from raw turns and metadata, crosscheck supplied CSV, inspect trajectory rate/units/available signals before designing train/test split. If unresolved, report exact missing link rather than assume identity.'))
    t=time.time()
    with openurl(LIST) as f:listing=json.load(f)
    assert 'nextPageToken' not in listing, 'Unexpected pagination; preserve listing and do not silently skip'
    dump(O/'OFFICIAL_OBJECT_LIST.json',listing)
    items=[i for i in listing['items'] if int(i['size'])>0]
    def fetch(item):
        relative=item['name'].removeprefix('behavior/');path=O/'ORIGINAL'/relative
        assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
        path.parent.mkdir(parents=True,exist_ok=True)
        url=f"https://storage.googleapis.com/{BUCKET}/{urllib.parse.quote(item['name'],safe='/')}?generation={item['generation']}"
        md=hashlib.md5();sha=hashlib.sha256();n=0
        with openurl(url) as source,path.open('xb') as out:
            while True:
                block=source.read(1024*1024)
                if not block:break
                out.write(block);n+=len(block);md.update(block);sha.update(block)
        assert n==int(item['size']),relative
        assert base64.b64encode(md.digest()).decode()==item['md5Hash'],relative
        print('VERIFIED',relative,n,flush=True)
        return dict(file=str(path),url=url,generation=item['generation'],bytes=n,md5=md.hexdigest(),sha256=sha.hexdigest(),official_md5_base64=item['md5Hash'])
    with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(fetch,items))
    dump(O/'DOWNLOAD_VERIFICATION.json',dict(files=records,total_files=len(records),total_bytes=sum(r['bytes'] for r in records),exit_code=0,wall_seconds=time.time()-t,model_load=0,body_step=0,neural_step=0))
    print('DOWNLOAD_COMPLETE',len(records),sum(r['bytes'] for r in records),flush=True)
if __name__=='__main__':main()
