"""One parent full trial or fresh saved-stream336 audit; no automatic retry."""
import argparse
import sys
import runtime as r

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('trial','audit'),required=True)
    parser.add_argument('--parent-resource-released',action='store_true')
    args = parser.parse_args()
    if not args.parent_resource_released:
        print('CODE_ONLY: no arrays/NN/Core/physics')
        return
    r.verify()
    preflight = r.read(r.HERE/'PREFLIGHT.json')
    if not preflight['all_passed'] or preflight['actual_Core_constructions']!=0 or preflight['actual_native_steps']!=0:
        raise ValueError('Exact zeroCore static preflight required')
    namespace,state = r.context(audit=args.mode=='audit')
    name = 'audit.py' if args.mode=='audit' else 'runner.py'
    tree,changes,routes = r.loop_tree(name)
    namespace.update(__name__='_rough_long_'+args.mode,__file__=str(r.LOOP/name))
    exec(compile(tree,str(r.LOOP/name)+'::long19p2','exec'),namespace)
    previous = list(sys.argv)
    sys.argv = [str(r.HERE/'run.py'),'--condition','V_empty','--parent-resource-released']
    try:
        namespace['main']()
    finally:
        sys.argv = previous

if __name__=='__main__':
    main()
