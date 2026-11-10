"""Execute collection, supervised training and physical assessment sequentially."""
import subprocess,sys,time,json
from pathlib import Path
script=Path(__file__).with_name('learn_stand.py')
for mode in ['collect','train','evaluate']:
 print(json.dumps(dict(stage=mode,status='starting')),flush=True)
 t=time.monotonic();subprocess.run([sys.executable,'-B',str(script),mode],check=True)
 print(json.dumps(dict(stage=mode,status='completed',wall_s=time.monotonic()-t)),flush=True)
