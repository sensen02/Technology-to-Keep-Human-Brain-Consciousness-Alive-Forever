from fit_body import OUT,OLD,load,ObservedBody,write,sha,torch,np,json
from record import BASE,WIDTH,read
from collections import Counter
def clock_matches(reference,observed,offset):
    total=Counter()
    for r in range(reference.shape[1]):
        source=np.flatnonzero(reference[:,r])
        detected=np.flatnonzero(observed[:,r])-offset
        i=j=match=0
        while i<len(source) and j<len(detected):
            d=detected[j]-source[i]
            if d< -2:j+=1
            elif d>2:i+=1
            else:i+=1;j+=1;match+=1
        total.update(matched=match,reference=len(source),called=len(detected))
    return total
def main():
    training=read(OUT/'training.json')
    assert training['new_sha']==sha(OUT/'weights.pt')!=training['source_sha']
    assert {n.split('_')[0] for n in training['episodes']}=={'10901','11311'}
    metrics=read(OUT/'evaluation.json')
    assert len(metrics)==7 and all('self_floor_travel_mm' in r['new'] for r in metrics)
    verify={}
    for seed in (11701,11903):
        rows=load(seed);dn_score=Counter();an_score=Counter()
        archive=BASE/'20261118T051014Z-neck-fault-assay/FORMAL'/f'seed{seed}'
        if seed==11903:archive=__import__('pathlib').Path('/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261118T052426Z-event-anchored-repair/FORMAL/seed11903')
        meta=read(archive/'INPUTS.json')
        with np.load(archive/'NEURAL.npz') as z:source=z['dn_emitted'];ups=z['an_emitted']
        for name,pos,f,dn,an in rows:
            idx=meta['modes'].index(name.split('_',1)[1])
            with np.load(OUT/'data'/name/'observed.npz') as z:
                dn_score+=clock_matches(source[:,idx],z['dn'],WIDTH-1)
                an_score+=clock_matches(ups[:,idx],z['an'],0)
            net=ObservedBody(OUT/'weights.pt').eval()
            with torch.no_grad():
                pred,_=net(pos[:-2],pos[1:-1],f[1:-1],dn[1:-1])
            error=float((pred-pos[2:]).norm(dim=-1).mean())
            expected=next(r for r in metrics if r['name']==name)['new']['step_mm']
            assert abs(error-expected)<1e-7
        verify[str(seed)]={'dn':dn_score,'an':an_score}
    write('verification.json',verify)
    print(verify)
if __name__=='__main__':main()
