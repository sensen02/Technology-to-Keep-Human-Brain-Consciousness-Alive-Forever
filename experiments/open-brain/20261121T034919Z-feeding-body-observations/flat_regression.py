from fit_feeding import OUT,BASE,PRE,OLD,ObservedBody,FlatFloor,torch,json,sys
sys.path.insert(0,str(BASE/'20261121T012712Z-body226-distal-cut'))
from fit_body import load
@torch.inference_mode()
def main():
    rows=[]
    for seed in [11701,11903]:
        name,p,f,dn,_=next(r for r in load(seed) if r[0].endswith(('intact','clean_on')))
        row=dict(name=name,teacher_net=float((p[-1,0,:2]-p[0,0,:2]).norm()))
        for label,body,pressure in [('parent',PRE,OLD),('adapted',OUT/'adapted',OUT/'adapted')]:
            net=ObservedBody(body/'weights.pt');floor=FlatFloor();floor.load_state_dict(torch.load(pressure/'floor.pt',weights_only=True))
            a,b=p[:1],p[1:2];last=floor(a,a);seq=[a,b]
            for t in range(1,len(p)-1):
                force=floor(a,b);step,_=net(a,b,torch.stack((last,force),1),dn[t:t+1])
                seq.append(step);a,b=b,step;last=force
            pred=torch.cat(seq)
            row[label]=dict(travel_mm=float((pred[-1,0,:2]-pred[0,0,:2]).norm()),endpoint_mm=float((pred[-1,0]-p[-1,0]).norm()),
                tail_rms_mm_s=float((pred[-41:,0,:2].diff(dim=0).square().sum(-1).mean()).sqrt()/.005))
        rows.append(row)
    (OUT/'flat-regression.json').write_text(json.dumps(rows,indent=2));print(rows)
if __name__=='__main__':main()
