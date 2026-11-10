"""Serial fixed six-condition gravity ladder; abort on execution failure."""
import subprocess,sys
from pathlib import Path
p=Path(__file__).with_name('joint_load.py')
for dt in ['.0005','.0000625']:
 for g in ['.25','.5','1.0']:
  subprocess.run([sys.executable,'-B',str(p),dt,g],check=True)
