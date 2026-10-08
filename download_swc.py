"""Download the FlyWire neuron skeletons (SWC, real 3D morphology, ~13 GB)."""
import os, sys, time, urllib.parse, urllib.request
ROOT=os.path.dirname(os.path.abspath(__file__))
tok=open(os.path.join(ROOT,'.flywire_api_token')).read().strip()
UA=("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
dest=os.path.join(ROOT,'data','flywire','fafb_skeletons_swc.zip')
url=("https://codex.flywire.ai/api/download_resource?"+urllib.parse.urlencode(
    {"data_product":"skeleton_swc_files","dataset":"fafb","api_token":tok}))
if os.path.exists(dest) and os.path.getsize(dest)>1e9:
    print("already present", os.path.getsize(dest)/1e9, "GB"); sys.exit(0)
req=urllib.request.Request(url, headers={"User-Agent":UA})
t0=time.time(); last=0
with urllib.request.urlopen(req, timeout=3600) as r:
    total=int(r.headers.get("Content-Length") or 0)
    print(f"Content-Length = {total/1e9:.2f} GB" if total else "no length", flush=True)
    done=0; tmp=dest+'.part'
    with open(tmp,'wb') as fh:
        while True:
            c=r.read(1<<22)
            if not c: break
            fh.write(c); done+=len(c)
            now=time.time()
            if now-last>20:
                last=now
                pct=f"{100*done/total:.1f}%" if total else "?"
                print(f"  {done/1e9:.2f} GB ({pct}) {done/1e6/max(now-t0,1):.1f} MB/s "
                      f"elapsed {now-t0:.0f}s", flush=True)
    os.replace(tmp,dest)
print(f"DONE {os.path.getsize(dest)/1e9:.2f} GB in {time.time()-t0:.0f}s")
