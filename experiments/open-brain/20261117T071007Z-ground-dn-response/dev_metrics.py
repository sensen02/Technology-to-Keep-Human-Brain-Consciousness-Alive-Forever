from pathlib import Path
import numpy as np
from ground_response import foot_excursions,case_for
for name in ('bluebell','web_b'):
 c=case_for(name,'dev')
 with np.load(Path('DEV_NATIVE')/f'{name}_dev_open'/'TRACE.npz') as data:
  for start in (c.action-3000,c.action-4200,c.action-4700):
   print(name,start,list(map(len,foot_excursions(data['feet'],(start,c.action-500)))))
  load=data['normal_load'][c.action+2600:c.action+5400]
  print('total min med max',load.sum(1).min(),np.median(load.sum(1)),load.sum(1).max())
