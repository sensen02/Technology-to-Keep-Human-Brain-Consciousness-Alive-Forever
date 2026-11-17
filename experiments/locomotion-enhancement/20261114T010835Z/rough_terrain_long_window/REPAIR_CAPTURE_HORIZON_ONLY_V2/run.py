"""V2 parent trial/audit entry; inherited loop source, isolated V2 runtime."""
import runtime as r

def main():
    program = r.load('_capture_v2_frozen_run',r.V1/'run.py')
    if program.r is not r:
        raise ValueError('V2 runtime routing required')
    program.main()

if __name__=='__main__':
    main()
