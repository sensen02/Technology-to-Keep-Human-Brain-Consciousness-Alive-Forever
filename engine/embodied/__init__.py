"""embodied -- a body, a scene, and a closed sensorimotor loop for the fly.

WHY THIS PACKAGE EXISTS
-----------------------
Every neural result in this project up to now was produced by a nervous system
with no body: a connectome with nowhere to act and nothing to sense.  The
approved plan therefore adds a physical body and a scene so that behaviour can
be produced and measured instead of assumed.

WHAT IS ADOPTED RATHER THAN REBUILT
-----------------------------------
The physics is NOT hand-written here.  It is FlyGym 2.1.0 / NeuroMechFly v2
(Apache-2.0, EPFL Ramdya lab) on MuJoCo 3.9.0, in a SEPARATE environment
(``venv_body``) so that installing MuJoCo cannot disturb the NEURON-based neural
suites in ``venv``.

UNITS (verified from the model, not assumed)
--------------------------------------------
The model is in MILLIMETRES.  ``mujoco_globals.yaml`` sets
``gravity: [0, 0, -9810]  # in mm/s^2`` and ``statistic.extent: 1  # mm``.
Therefore in every record produced by this package:
    length mm      time s      angle rad      gravity mm/s^2      mass kg
A fly thorax height is about 1 mm, which is what the model reports; treating
those numbers as metres would overstate every distance by 1000x.  This module
converts NOTHING silently: values are passed through in model units and the unit
string is written next to every recorded quantity.

Honesty boundary
----------------
A body that walks is not a claim about experience, and a controller that makes
it walk is an ENGINEERING BASELINE, not evidence that a connectome generates
behaviour.  Anything produced by a hand-written controller is labelled as such
in the report.
"""
from __future__ import annotations

from .body_backend import (
    MM, MM_PER_S, RAD, SECONDS, GEOMETRY_NOTE, GL_BACKENDS,
    BodyBackend, BodyConfig, BodyObservation, select_gl_backend, summarise_walk,
)

__all__ = ["BodyBackend", "BodyConfig", "BodyObservation",
           "summarise_walk", "select_gl_backend", "GL_BACKENDS",
           "MM", "MM_PER_S", "RAD", "SECONDS", "GEOMETRY_NOTE"]
