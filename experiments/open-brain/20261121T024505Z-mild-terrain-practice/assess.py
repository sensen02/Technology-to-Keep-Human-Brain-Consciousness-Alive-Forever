from train_practice import *
@torch.inference_mode()
def evaluate():
    flat_trials,terrain,_=corpus('test')
    rows=flat_trials+terrain
    floor=floor_model()
    pos,touch,dn=[torch.stack([r[i] for r in rows]) for i in (1,2,3)]
    results=[dict(episode=r[0],teacher_net_displacement_mm=float((p[-1,0,:2]-p[0,0,:2]).norm()),
                  teacher_tail_net100_mm_s=float((p[-1,0]-p[-21,0]).norm()/.1)) for r,p in zip(rows,pos)]
    for name,path in [('previous',PARENT),('flat_control',OUT/'flat_control'),('expanded',OUT/'expanded')]:
        net=ObservedBody(path/'weights.pt').eval()
        for i,row in enumerate(results):
            pred=net(pos[i,:-2],pos[i,1:-1],touch[i,1:-1],dn[i,1:-1])[0]
            row[name]={'step_mm':float((pred-pos[i,2:]).norm(dim=-1).mean())}
        for use_floor in (False,True):
            seq=[pos[:,0],pos[:,1]]
            last=floor(seq[0],seq[0])
            for k in range(1,399):
                est=floor(seq[-2],seq[-1])
                loads=torch.stack((last,est),dim=1) if use_floor else touch[:,k]
                step,_=net(seq[-2],seq[-1],loads,dn[:,k])
                seq.append(step);last=est
            trajectory=torch.stack(seq,dim=1)
            assert torch.isfinite(trajectory).all()
            for i,p in enumerate(trajectory):
                prefix='self' if use_floor else 'observed'
                metrics=results[i][name]
                metrics[prefix+'_net_displacement_mm']=float((p[-1,0,:2]-p[0,0,:2]).norm())
                metrics[prefix+'_endpoint_mm']=float((p[-1,0]-pos[i,-1,0]).norm())
                metrics[prefix+'_tail_net100_mm_s']=float((p[-1,0]-p[-21,0]).norm()/.1)
                metrics[prefix+'_tail_rms200_mm_s']=float((p[-41:,0,:2].diff(dim=0).square().sum(-1).mean()).sqrt()/.005)
        for h in [10,40]:
            start=torch.arange(1,400-h,h)
            a,b=(pos[:,start-1].flatten(0,1),pos[:,start].flatten(0,1))
            for j in range(h):a,b=b,net(a,b,touch[:,start+j].flatten(0,1),dn[:,start+j].flatten(0,1))[0]
            err=(b.reshape(len(rows),len(start),21,3)-pos[:,start+h]).norm(dim=-1).mean((1,2))
            for row,v in zip(results,err):row[name][f'roll{h*5}_mm']=float(v)
    write(OUT/'evaluation.json',results)
if __name__=='__main__':evaluate()
