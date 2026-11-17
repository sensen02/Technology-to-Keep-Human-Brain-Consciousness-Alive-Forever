"""Parent-only unchanged336 source-neck loop on one declared physical terrain."""
import argparse
from adapter import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('plane_control','flat_box_control','mild_blocks_seed17'),required=True)
    parser.add_argument('--parent-resource-released',action='store_true')
    args = parser.parse_args()
    if not args.parent_resource_released:
        print('SOURCE_ONLY no model load or integration')
        return
    run('runner.py',args.case)


if __name__ == '__main__':
    main()
