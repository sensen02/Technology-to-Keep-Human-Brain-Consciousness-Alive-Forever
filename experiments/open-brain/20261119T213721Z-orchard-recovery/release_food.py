"""Release the local tarsal feeding posture when measured intake meets the meal target."""
import json
from context import tr
import live_food

def run():
    original=live_food.active_build
    def build(crop):
        env,bank=original(crop)
        received=0.;advance=bank.advance
        def measure(*args,**kwargs):
            nonlocal received
            receipt=advance(*args,**kwargs)
            received+=receipt.ingested.volume
            return receipt
        bank.advance=measure
        neutral=env.task.before_step
        base_fit=live_food.place
        def place(*args,**kwargs):
            q=base_fit(*args,**kwargs)
            # live_food installs its sensory hook immediately after returning from placement.
            # Wrap environment.step so it can release only after an actual feeding receipt.
            step=env.step
            def perform(action):
                if received>=tr.docking.MEAL:
                    env.task.before_step=neutral
                return step(action)
            env.step=perform
            return q
        live_food.place=place
        return env,bank
    live_food.active_build=build
    live_food.run()
if __name__=='__main__':run()
