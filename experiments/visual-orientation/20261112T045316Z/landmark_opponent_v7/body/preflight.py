"""Parent actual0Core newneuralreceptor/loader and unchanged far optical/control preflight."""
from adapter import entry_namespace
from runtime import *

def main():
 ns=entry_namespace('preflight.py');original=ns['save']
 def save_with_receptor_checks(path,value):
  if Path(path).name=='PREFLIGHT.json':
   visual,motor,vp,mp=load_policies()
   import torch
   extra=dict(actual_new_opponent_architecture=visual.identity()['format']=='VISION_FIXED_NEURAL_OPPONENT_ARCH_V7',fixed_neural_receptor_weight_exact=torch.equal(visual.opponent_weight,torch.tensor([1.,-.5,-.5],dtype=torch.float64).reshape(1,3,1,1)),fixed_neural_receptor_bias_zero=visual.opponent_bias.item()==0,first_learnedconv_singlechannel=visual.conv1.in_channels==1,only_fourteen_trainable_tensors=len(list(visual.parameters()))==14,actual_opponent_format_final1000=vp['format_version']=='VISION_OPPONENT_POLICY_V7' and vp['step']==1000)
   if not all(extra.values()):raise ValueError('actual new receptor identity failed')
   value['checks'].update(extra);value['all_passed']=all(value['checks'].values());value['model_loads']=4;value['new_receptor_identity_checked']=True
  original(path,value)
 ns['save']=save_with_receptor_checks;ns['main']()
if __name__=='__main__':main()
