"""Separate-process scalar feature/metric/gradient audit; no new model fitting."""
from pathlib import Path
import csv,json,math,hashlib
import numpy as np
S=Path('/home/sensen/Desktop/engine修复验收/s4-banc-individual-behavior/20261101T211209Z')
O=Path('/home/sensen/Desktop/engine修复验收/s4-banc-causal-baselines/20261101T212622Z')
def main():
    events=list(csv.DictReader((S/'TURN_EVENTS.csv').open()));z=np.load(O/'FEATURES_AND_PREDICTIONS.npz');r=json.loads((O/'RESULT.json').read_text())
    means=float(z['previous_interval_mean']);sd=float(z['previous_interval_sd']);X=[];Y=[]
    for i in z['event_indices']:
        i=int(i);assert i>=3 and events[i]['included_in_upstream_rBias']=='1'
        e=events[i];past=[events[i-j] for j in (1,2,3)];timeprev=float(past[0]['time_float64_s']);assert timeprev<float(e['time_float64_s'])
        interval=timeprev-float(past[1]['time_float64_s'])
        X.append([1.,*[2*int(v['right'])-1 for v in past],float(int(e['from_arm'])==1),float(int(e['from_arm'])==2),(math.log1p(interval)-means)/sd]);Y.append(int(e['right']))
    X=np.asarray(X);Y=np.asarray(Y);assert np.max(abs(X-z['X']))<1e-12 and np.array_equal(Y,z['y'])
    part=z['split'];train=part==0;valid=part==1;test=part==2
    rawlog=np.array([math.log1p(float(events[int(i)-1]['time_float64_s'])-float(events[int(i)-2]['time_float64_s'])) for i in z['event_indices']])
    assert abs(rawlog[train].mean()-means)<1e-12 and abs(rawlog[train].std()-sd)<1e-12
    assert abs(float(Y[train].mean())-r['train_constant'])<1e-15
    gradients=[]
    for lam,w,p in zip(z['lambdas'],z['all_weights'],z['all_predictions']):
        prediction=np.array([1/(1+math.exp(-float(sum(v*t for v,t in zip(row,w))))) for row in X]);assert abs(prediction-p).max()<1e-12
        gradient=np.mean(X[train]*(prediction[train]-Y[train])[:,None],axis=0);gradient[1:]+=lam*w[1:]/train.sum();gradients.append(float(abs(gradient).max()));assert abs(gradient).max()<1e-5
    chosen=min(r['lambda_fits'],key=lambda a:(a['validation']['NLL'],-a['lam']))['lam'];assert chosen==r['selected_lambda']
    assert np.array_equal(z['selected_weights'],z['all_weights'][list(z['lambdas']).index(chosen)])
    for model,key in [('train_constant','constant'),('previous_turn_markov','markov'),('selected_logistic','selected_logistic')]:
        p=z[key]
        for splitname,mask in [('train',train),('validation',valid),('test',test)]:
            yy=Y[mask];pp=p[mask];n=len(pp)
            loss=math.fsum([-math.log(float(v)) if y else -math.log1p(-float(v)) for y,v in zip(yy,pp)])/n
            brier=math.fsum([(float(v)-int(y))**2 for y,v in zip(yy,pp)])/n
            assert abs(loss-r['metrics'][model][splitname]['NLL'])<1e-12 and abs(brier-r['metrics'][model][splitname]['Brier'])<1e-12
    yy=Y[test];p=z['selected_logistic'][test];q=z['constant'][test]
    diff=np.array([(-math.log(float(a))+math.log(float(b))) if y else (-math.log1p(-float(a))+math.log1p(-float(b))) for y,a,b in zip(yy,p,q)])
    assert abs(diff.mean()-r['paired_test_NLL_difference_logistic_minus_constant'])<1e-12
    rng=np.random.default_rng(7311);boot=[];n=len(diff)
    for _ in range(1000):
        starts=rng.integers(0,n,size=(n+15)//16);indices=[(int(s)+j)%n for s in starts for j in range(16)][:n];boot.append(math.fsum(float(diff[j]) for j in indices)/n)
    assert abs(np.array(boot)-z['bootstrap_means']).max()<1e-12
    for f,h in r['source_sha256'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
    result=dict(passed=True,feature_max_error=float(abs(X-z['X']).max()),gradient_maxima=gradients,selected_lambda=chosen,all_metrics_reconstructed=True,bootstrap_reconstructed=True,model_fit_calls=0,neural_step=0,body_step=0,limits='Checks observational teacher-forced predictors only; no neural or embodied causality evidence.')
    (O/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
