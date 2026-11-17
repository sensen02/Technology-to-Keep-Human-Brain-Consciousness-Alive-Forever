"""One namespace layer, exact originaldirectloops and numericalconstants, newmetadata globals."""
from pathlib import Path
import ast
HERE=Path(__file__).resolve().parent;OLD_BODY=Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/neck-real-structure/20261113T045302Z/sensory_body')
NAMES=('runner.py','audit.py','preflight.py','summary.py','review.py','orchestrate.py')
def tree(name):
 if name not in NAMES:raise ValueError('Unregisteredimmutableentry')
 result=ast.parse((OLD_BODY/name).read_text());ast.fix_missing_locations(result);return result
def namespace(name):
 import runtime
 ns=dict(__name__='_sensoryrepair_same_'+name.replace('.','_'),__file__=str(HERE/name));exec(compile(tree(name),str(HERE/name)+'::exactvalidatedoriginaldirectloop','exec'),ns);ns.update({k:v for k,v in runtime.__dict__.items() if not k.startswith('_')});return ns
def run(name):namespace(name)['main']()
