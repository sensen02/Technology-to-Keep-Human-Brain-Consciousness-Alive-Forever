"""Read-only conditioning audit: no optimizer, no physics, no saved candidate."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent.parent
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from fast_motor import load,sha256,verify_hashes
from fast_data import prepare

CK=ROOT/'continuation3000/TRAIN/checkpoint_003000.pt'
SHA='17fd103492007dc4de4ae9832fa39e4098cef980c4bebb03425c99f7c2f093a8'
GROUPS={'physical112':(0,112),'base_hidden128':(112,240),'normalized_command2':(240,242)}
def values(t):return t.detach().cpu().tolist()
def summary(t):
 t=t.detach().flatten();return dict(min=float(t.min()),median=float(t.median()),max=float(t.max()),rms=float(t.square().mean().sqrt()))
def layer_stats(pre):
 a=pre.tanh();return dict(preactivation=summary(pre),abs_activation_ge095=float((a.abs()>=.95).double().mean()),abs_activation_ge099=float((a.abs()>=.99).double().mean()),mean_tanh_derivative=float((1-a.square()).mean()))
def spread(v):
 active=v[v>1e-24];return dict(active_features=len(active),zero_variance_features=int((v<=1e-24).sum()),positive_variance_min=float(active.min()),positive_variance_max=float(active.max()),positive_variance_ratio=float(active.max()/active.min()),p95_p05_ratio=float(torch.quantile(active,.95)/torch.quantile(active,.05)))

def main():
 torch.set_num_threads(1);torch.manual_seed(20261110)
 model,payload=load(CK,SHA)
 frozen={**payload['config']['sources'],**payload['config']['data_hashes'],str(CK):SHA,str(HERE/'analyze.py'):sha256(HERE/'analyze.py')}
 verify_hashes(frozen)
 episodes,data=prepare(model)
 F=torch.cat([e['features'] for e in episodes]);G=torch.cat([e['gate'] for e in episodes]);M=torch.cat([e['mean'] for e in episodes]);Y=torch.cat([e['y'] for e in episodes])
 train=torch.cat([e['features'][960:] for e in episodes]);mu=train.mean(0);std=train.std(0,unbiased=False);scale=std.clamp_min(.01)
 W=model.encoder.weight.detach().clone();b=model.encoder.bias.detach().clone()
 Wn=(W*scale).requires_grad_(True);bn=(b+W@mu).requires_grad_(True)
 f=F.clone().requires_grad_(True)
 pre=model.encoder(f);hpre=model.hidden(pre.tanh());out=G*model.head(hpre.tanh()).tanh()
 fn=F.clone().requires_grad_(True);z=(fn-mu)/scale
 pren=torch.nn.functional.linear(z,Wn,bn);hpren=model.hidden(pren.tanh());outn=G*model.head(hpren.tanh()).tanh()
 # Every saved row/all outputs; original-input gradients of sum(output^2) test full mapping.
 gx=torch.autograd.grad(out.square().sum(),f,retain_graph=True)[0]
 gxn=torch.autograd.grad(outn.square().sum(),fn,retain_graph=True)[0]
 output_error=float((out-outn).abs().max());pre_error=float((pre-pren).abs().max());grad_error=float((gx-gxn).abs().max())
 check=dict(all11520_rows=len(F)==11520,all_predictions_le1e12=output_error<=1e-12,all_original_input_gradients_le1e12=grad_error<=1e-12,preactivation_le1e12=pre_error<=1e-12)
 # Loss and parameter gradients are measured without applying an update.
 mask=torch.cat([torch.arange(2880)>=960 for e in episodes]);target=Y[mask]
 loss=((M[mask]+out[mask]).tanh()-target).square().mean()
 lossn=((M[mask]+outn[mask]).tanh()-target).square().mean()
 gw,gb=torch.autograd.grad(loss,[model.encoder.weight,model.encoder.bias],retain_graph=True)
 gwn,gbn=torch.autograd.grad(lossn,[Wn,bn],retain_graph=True)
 check['loss_identical_le1e12']=abs(float(loss-lossn))<=1e-12
 # Chain rule transforms normalized-parameter gradients back to original parameters.
 mapped_gw=gwn*scale+gbn[:,None]*mu
 check['parameter_gradient_chain_rule_le1e12']=float((gw-mapped_gw).abs().max())<=1e-12 and float((gb-gbn).abs().max())<=1e-12
 group={};contrib=[]
 centered=train-mu
 for name,(a,c) in GROUPS.items():
  part=centered[:,a:c]@W[:,a:c].T;contrib.append(part)
  group[name]=dict(feature_mean=values(mu[a:c]),feature_std=values(std[a:c]),feature_min=values(train[:,a:c].min(0).values),feature_max=values(train[:,a:c].max(0).values),feature_abs_rms=values(train[:,a:c].square().mean(0).sqrt()),std_summary=summary(std[a:c]),abs_value=summary(train[:,a:c].abs()),encoder_weight_frobenius=float(W[:,a:c].norm()),centered_encoder_contribution_variance_sum=float(part.var(0,unbiased=False).sum()),centered_encoder_contribution_rms=float(part.square().mean().sqrt()),input_gradient_L2=float(gx[:,a:c].norm()),input_gradient_per_scalar_rms=float(gx[:,a:c].square().mean().sqrt()))
 total=centered@W.T
 V=torch.stack(contrib)
 cross={}
 for i,ni in enumerate(GROUPS):
  for j,nj in enumerate(GROUPS):
   ai=V[i].flatten();aj=V[j].flatten();cross[ni+'__'+nj]=float(torch.dot(ai,aj)/(ai.norm()*aj.norm()).clamp_min(1e-30))
 cov=centered.T@centered/len(centered);denom=std[:,None]*std[None,:];corr=torch.where(denom>1e-24,cov/denom.clamp_min(1e-24),torch.zeros_like(cov));upper=torch.triu(torch.ones_like(corr,dtype=torch.bool),1)
 ps={}
 offset=0
 for e in episodes:
  q=F[offset:offset+2880];r=model.encoder(q);rr=model.hidden(r.tanh());ps[e['name']]=dict(encoder_all=layer_stats(r),encoder_postprime=layer_stats(r[960:]),hidden_postprime=layer_stats(rr[960:]));offset+=2880
 report=dict(status='PASS' if all(check.values()) else 'FAIL',checks=check,checkpoint=str(CK),checkpoint_sha256=SHA,rows=len(F),training_statistics_rows=len(train),statistics_scope='only qualified original4teacher postprime train rows; mean/std shared across4; no mixed224/225 data',std_floor=.01,std_floor_features=int((std<.01).sum()),groups=group,encoder_postprime=layer_stats(model.encoder(train)),hidden_postprime=layer_stats(model.hidden(model.encoder(train).tanh())),group_centered_contribution_cosines=cross,total_centered_encoder_variance_sum=float(total.var(0,unbiased=False).sum()),sum_group_variance=float(sum(q['centered_encoder_contribution_variance_sum'] for q in group.values())),encoder_constant_offset=dict(raw_Wmean_norm=float((W@mu).norm()),bias_norm=float(b.norm()),combined_mean_preactivation_norm=float((b+W@mu).norm())),feature_correlations=dict(abs_pair_median=float(corr[upper].abs().median()),abs_pair_p95=float(torch.quantile(corr[upper].abs(),.95)),abs_pair_over095_count=int((corr[upper].abs()>.95).sum())),variance_diagonal_before=spread(std.square()),variance_diagonal_after=spread((std/scale).square()),function_preserving_algebra=dict(formula='z=(f-mu)/s; Wn=W*s; bn=b+W@mu; all other neural weights/gate/base unchanged',prediction_max_abs_error=output_error,preactivation_max_abs_error=pre_error,original_input_gradient_max_abs_error=grad_error,parameter_gradient_chain_rule_max_abs_error=float((gw-mapped_gw).abs().max()),loss_original=float(loss),loss_standardized=float(lossn)),encoder_parameter_gradient=dict(original_weight_norm=float(gw.norm()),standardized_weight_norm=float(gwn.norm()),original_bias_norm=float(gb.norm()),standardized_bias_norm=float(gbn.norm()),original_weight_abs_summary=summary(gw.abs()),standardized_weight_abs_summary=summary(gwn.abs()),original_per_column_L2=values(gw.norm(dim=0)),standardized_per_column_L2=values(gwn.norm(dim=0))),per_case=ps,source_and_data_hashes=frozen,physical_integration_steps=0,optimizer_steps=0,saved_candidate_checkpoints=0,limitations=['Teacher-distribution diagnostic only; no causal claim for actual slip.','Function-preserving normalization leaves all current saturation and action values unchanged; improved parameter conditioning does not prove better fitting.','No whitening; correlations remain, floored low-variance/constant features are not unit variance.','Adam optimizer moments cannot simply be retained under changed parameter coordinates without explicitly justified transformation; a new controlled training comparison must declare its optimizer policy.'])
 verify_hashes(frozen)
 (HERE/'DIAGNOSIS.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:report[k] for k in ('status','checks','encoder_postprime','hidden_postprime','variance_diagonal_before','variance_diagonal_after','function_preserving_algebra','encoder_parameter_gradient') if k!='encoder_parameter_gradient'},indent=2))
 if not all(check.values()):raise ValueError('normalization equivalence check failed')
if __name__=='__main__':main()
