"""One gray-world trial or fresh actual-RGB fourNN/source audit, parent only."""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('trial','audit'),required=True)
    p.add_argument('--parent-resource-released',action='store_true')
    a = p.parse_args()
    if not a.parent_resource_released:
        print('SOURCE_ONLY no execution')
        return
    import runtime
    runtime.verify()
    proof = runtime.read(HERE/'PREFLIGHT.json')
    if not proof['all_passed'] or not all(proof['checks'].values()):
        raise ValueError('Actual compiled physical equality and gray1Core preflight required')
    namespace,state = runtime.context(audit=a.mode=='audit')
    adapter = runtime.load('_gray_frozen_loop_adapter',runtime.PARENT/'terrain_neural_baseline/adapter.py')
    name = 'runner.py' if a.mode=='trial' else 'audit.py'
    tree,changes = adapter.tree(name)
    namespace.update(__name__='_gray_visual_'+a.mode,__file__=str(adapter.LOOP/name))
    exec(compile(tree,str(adapter.LOOP/name)+'::gray_world_color_only','exec'),namespace)
    before = list(sys.argv)
    sys.argv = [str(HERE/'run.py'),'--condition','V_empty','--parent-resource-released']
    try:
        namespace['main']()
    finally:
        sys.argv = before


if __name__=='__main__':
    main()
