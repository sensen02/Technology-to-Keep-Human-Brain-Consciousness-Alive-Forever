"""S155 saved-data independent audit against S154, no simulation."""
from pathlib import Path
import argparse,types,sys
ROOT=Path(__file__).resolve().parent
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--output',type=Path,required=True);p=pa.parse_args().output
 src=(ROOT/'verify_activation_rate_probe.py').read_text();src=src.replace("s149-half-gain-physical/20261024T213108429114Z","s154-activation-rate/20261024T215751775489Z")
 src=src.replace("expected[:,0]==.0005","expected[:,0]==.001").replace("expected[:,0]=.001","expected[:,0]=.002").replace("ACTIVATION_RATE_INDEPENDENT_CHECK.json","ACTIVATION_DOSE_INDEPENDENT_CHECK.json")
 # S154 checker transforms historical generic audit source strings; transform replacement text, not old-match baseline.
 mod=types.ModuleType('_s155_independent');mod.__file__=str(Path(__file__));exec(compile(src,str(Path(__file__)),'exec'),mod.__dict__)
 old=sys.argv
 try:sys.argv=[str(Path(__file__)),'--output',str(p)];mod.main()
 finally:sys.argv=old
if __name__=='__main__':main()
