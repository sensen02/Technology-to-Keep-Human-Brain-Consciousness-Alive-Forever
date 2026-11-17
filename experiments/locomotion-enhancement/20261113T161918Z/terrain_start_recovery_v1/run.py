"""Parent-only one repaired scene trial OR fresh saved-RGB fourNN audit."""
from pathlib import Path
import argparse
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('trial','audit'),required=True)
    parser.add_argument('--parent-resource-released',action='store_true')
    args = parser.parse_args()
    if not args.parent_resource_released:
        print('SOURCE_ONLY no model/NN/array/physical execution')
        return
    import runtime
    runtime.verify()
    if not runtime.read(HERE/'PREFLIGHT.json')['all_passed']:
        raise ValueError('Actual new-scene oneCore preflight required')
    namespace,state = runtime.context(audit=args.mode=='audit')
    adapter = runtime.load('_scene_v1_frozen_AST_adapter',runtime.OLD_NEURAL/'adapter.py')
    name = 'audit.py' if args.mode=='audit' else 'runner.py'
    tree,changes = adapter.tree(name)
    namespace.update(__name__='_scene_start_repair_'+args.mode,__file__=str(adapter.LOOP/name))
    exec(compile(tree,str(adapter.LOOP/name)+'::fixed_start_area_scene_v1','exec'),namespace)
    previous = list(sys.argv)
    sys.argv = [str(HERE/'run.py'),'--condition','V_empty','--parent-resource-released']
    try:
        namespace['main']()
    finally:
        sys.argv = previous


if __name__ == '__main__':
    main()
