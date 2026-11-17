"""Opt-in dry claw contact on named rough surfaces, keeping flat-ground triad contact."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from hook import HERE, LEGS

FOOT_LAYER = 4
ROUGH_LAYER = 8
EXPECTED = {f"{leg}Tarsus5_claw_{side}_{link}" for leg in LEGS
            for side in range(2) for link in range(3)}


def number(geom, key):
    # This asset explicitly assigns the rough-box masks, even when the floor is implicit.
    return int(geom.get(key, "1"))


def build(source, destination, rough_names):
    tree = ET.parse(source)
    geoms = {geom.get("name"): geom for geom in tree.findall(".//worldbody//geom")}
    if {name for name in geoms if name and "_claw_" in name} != EXPECTED:
        raise ValueError("Expected the complete set of six dual-claw feet")
    if any((number(geom, "contype") | number(geom, "conaffinity")) & (FOOT_LAYER | ROUGH_LAYER)
           for geom in geoms.values()):
        raise ValueError("Collision layers already assigned")
    world = {geom.get("name"): geom for geom in tree.findall("./worldbody/geom")}
    for name in rough_names:
        surface = world.get(name)
        if surface is None or surface.get("type") != "box":
            raise ValueError(f"Rough contact requires an explicit world box: {name}")
        surface.set("contype", str(number(surface, "contype") | ROUGH_LAYER))
        surface.set("conaffinity", str(number(surface, "conaffinity") | FOOT_LAYER))
    for name in EXPECTED:
        geoms[name].set("contype", str(FOOT_LAYER))
        geoms[name].set("conaffinity", str(ROUGH_LAYER))
    with destination.open("xb") as stream:
        tree.write(stream, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, default=HERE / "BODY_CLAWS_EXPERIMENTAL.xml")
    parser.add_argument("--rough-box", action="append", default=[])
    args = parser.parse_args()
    build(args.source, args.output, set(args.rough_box))


if __name__ == "__main__":
    main()
