import subprocess,sys
from pathlib import Path
for bias in ['0','.02']:
 for vel in ['0','-.1','.1']:
  subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('sensory_bias_body.py')),bias,vel],check=True)
