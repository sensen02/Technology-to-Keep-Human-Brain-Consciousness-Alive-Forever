"""Fresh-process loading, whole-episode reference evaluation and predictions."""
import argparse
from shared import *
from encoder import load,Encoder


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('case',choices=TRAIN_CASES+HELD_CASES)
    p.add_argument('output',type=Path);args=p.parse_args();torch.set_num_threads(1)
    fit=read(HERE/'FIT_RESULT.json');assert sha(fit['candidate'])==fit['candidate_sha256']
    verify(fit['source_sha256']);arrays,meta=load_data(args.case)
    model,saved=load(fit['candidate']);initial,_=load(HERE/'FIT/update000.pt')
    x=torch.from_numpy(arrays['features_hz']).double();y=arrays['teacher_rate_hz']
    prediction=model(x).numpy();baseline=initial(x).numpy()
    mse=float(np.mean((prediction-y)**2));old=float(np.mean((baseline-y)**2))
    nrmse=(np.mean((prediction-y)**2,axis=0)/np.mean(y*y,axis=0))**.5
    z=torch.zeros_like(x[:100]);zero=bool((model(z)==0).all())
    probe=z.clone();probe[:,0]=100;isolated=bool((model(probe)[:,1]==0).all())
    gates=dict(loss_improved=mse<=.8*old,each_source_nrmse=bool((nrmse<=.20).all()),
               silence_zero=zero,channel_isolation=isolated,finite=bool(np.isfinite(prediction).all()),
               bounded=bool(((prediction>=0)&(prediction<=CAP_HZ)).all()))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'PREDICTIONS.npz',prediction=prediction,initial=baseline,target=y,
                        time_s=arrays['time_s'])
    write(args.output/'RESULT.json',dict(case=args.case,pass_=all(gates.values()),gates=gates,
           mse_hz2=mse,initial_mse_hz2=old,relative_reduction=1-mse/old,nrmse=nrmse.tolist(),
           checkpoint_sha256=fit['candidate_sha256'],data_sha256=meta['trace_sha256'],
           predictions_sha256=sha(args.output/'PREDICTIONS.npz'),source_sha256=sources({fit['candidate'],HERE/'FIT_RESULT.json'}),
           scope='engineering input-reference agreement; not autonomous behavior'))
    print(args.case,gates,'NRMSE',nrmse.tolist(),'reduction',1-mse/old,flush=True)


if __name__=='__main__':main()
