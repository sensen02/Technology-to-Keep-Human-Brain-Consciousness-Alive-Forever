"""End-to-end HTTP checks against running project service; no fabricated measurements."""
import json,urllib.request,urllib.error
from pathlib import Path
BASE='http://127.0.0.1:8766';ROOT=Path(__file__).resolve().parent
def req(path,data=None,origin=None):
    h={}
    if data is not None:h['Content-Type']='application/json'
    if origin:h['Origin']=origin
    q=urllib.request.Request(BASE+path,data=json.dumps(data).encode() if data is not None else None,headers=h)
    try:
        with urllib.request.urlopen(q,timeout=15) as r:return r.status,r.read()
    except urllib.error.HTTPError as e:return e.code,e.read()
def main():
    checks=[]
    def check(n,c):assert c,n;checks.append(n)
    code,b=req('/api/status');check('backend health',code==200 and json.loads(b)['service']=='cell-workbench')
    code,b=req('/api/data');d=json.loads(b);check('backend real geometry',code==200 and len(d['neural']['ids'])==6000)
    check('fixed geometry',d['electrodes']['diameter_um']==7 and d['electrodes']['pitch_um']==20)
    check('private paths blocked',req('/../bench/config.template.json')[0]==404)
    check('unknown api rejected',req('/api/nonexistent')[0]==404)
    check('cross origin write blocked',req('/api/jobs/recording',{},'http://evil.invalid')[0]==403)
    check('invalid bench rejected',req('/api/bench/evaluate',{'config':{},'csv':''})[0]==400)
    config=json.loads((ROOT/'bench/config.template.json').read_text());csv=(ROOT/'bench/measurements.template.csv').read_text()
    code,b=req('/api/bench/evaluate',{'config':config,'csv':csv});report=json.loads(b)
    check('empty real data NOT_RUN',code==200 and report['status']=='NOT_RUN' and not report['readiness_claim'])
    check('persisted bench via status',json.loads(req('/api/status')[1])['bench']['status']=='NOT_RUN')
    code,b=req('/api/jobs/recording',{});check('real recording job accepted',code==202)
    check('single job guard',req('/api/jobs/recording',{})[0]==409)
    print(json.dumps({'passed':len(checks),'checks':checks,'job':json.loads(b)},indent=2))
if __name__=='__main__':main()
