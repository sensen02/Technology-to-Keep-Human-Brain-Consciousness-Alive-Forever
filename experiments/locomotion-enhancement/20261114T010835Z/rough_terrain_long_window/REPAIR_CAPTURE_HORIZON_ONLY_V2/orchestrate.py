"""One corrected V2 longcase + fresh336 audit; never rerun archived V1."""
import runtime as r

def main():
    program = r.load('_capture_v2_frozen_orchestrate',r.V1/'orchestrate.py')
    if program.r is not r:
        raise ValueError('V2 runtime routing required')
    program.main()

if __name__=='__main__':
    main()
