"""Enable native mouth joints on the derived diagnostics fixture only."""
from unittest.mock import patch
import numpy as np
from context import tr
from flybody.fly_envs import fruitfly

def active_build(crop):
    native=fruitfly.FruitFly
    def make(*args,**kwargs):
        if kwargs.get('name')=='walker':
            kwargs['use_mouth']=True
        return native(*args,**kwargs)
    with patch.object(fruitfly,'FruitFly',make):
        env,bank=original_build(crop)
    return env,bank

original_build=tr.build
