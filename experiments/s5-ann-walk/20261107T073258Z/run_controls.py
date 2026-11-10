"""Predeclared matched physical controls for the new replacement ANN."""
from neural_walk import OUT, simulate, prereg

if __name__ == '__main__':
    prereg()
    path=OUT/'MODEL.npz'
    for tag,seconds,mode,disturb in [
        ('hold_4s',4.,'hold',False),
        ('motor_cut_4s',4.,'motor_cut',False),
        ('teacher_11s',11.,'teacher',False),
        ('ann_disturb_11s',11.,'ann',True),
    ]:
        simulate(path,tag,seconds,mode,disturb)
