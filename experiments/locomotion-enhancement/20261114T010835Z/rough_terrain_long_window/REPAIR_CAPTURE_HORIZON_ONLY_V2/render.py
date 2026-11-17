"""V2 parent saved actualRGB/body renderer; inherited view, V2 identity only."""
import runtime as r

def main():
    program = r.load('_capture_v2_frozen_render',r.V1/'render.py')
    if program.r is not r:
        raise ValueError('V2 runtime routing required')
    program.main()

if __name__=='__main__':
    main()
