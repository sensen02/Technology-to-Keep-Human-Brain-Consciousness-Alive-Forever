"""MEADOW: the grass-stand environment for the multi-camera arena.

WHAT THIS ADDS TO THE EXISTING NATURAL SCENE
--------------------------------------------
``engine/embodied/natural_scene.py`` already does fBm micro-relief, a CC0 ground
photograph, a gradient sky and rebalanced lighting.  It does NOT do grass.  This module
supplies the three things that make the ground a grass STAND rather than a rough plane:

  1. a GRASS HEIGHTFIELD with the geometry of blades (``grass_scene``), whose
     statistics are audited rather than asserted;
  2. a real CC0 PHOTOGRAPH of mown lawn as the surface colour, cached on disk with a
     provenance sidecar so it can be re-fetched and legally reused;
  3. the combination as a scene preset (``meadow_grass``) that the body backend can
     apply like any other preset.

It is a separate module for the same reason ``natural_scene.py`` is separate from
``scene.py``: neither file has to be edited when the other grows.  The integration
point in ``natural_scene`` is ONE branch in ``_build_terrain``.

THE LICENCE IS PART OF THE ASSET
--------------------------------
``GRASS_CC0_ASSET`` in ``grass_scene`` records the source page, the author, the date and
the licence, and :func:`ensure_grass_texture` refuses to use a cached file whose
recorded sha256 does not match, so a swapped file cannot silently inherit the licence of
the original.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from grass_scene import GRASS_CC0_ASSET, generate_grass_heightfield

__all__ = ["ensure_grass_texture", "grass_texture_provenance",
           "GRASS_TEXTURE_PNG", "MEADOW_PRESET_NAME", "meadow_scene_config",
           "refuse_unverified_texture"]

HERE = Path(__file__).resolve().parent
ASSET_DIR = HERE / "outputs" / "embodied_body" / "assets"
SRC_JPG = ASSET_DIR / GRASS_CC0_ASSET["file"]
GRASS_TEXTURE_PNG = ASSET_DIR / "grass_meadow_photo.png"
SIDECAR = ASSET_DIR / "grass_meadow_photo.provenance.json"
MEADOW_PRESET_NAME = "meadow_grass"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def refuse_unverified_texture(path: Path) -> None:
    """Raise unless ``path`` matches the recorded sha256 of the CC0 asset.

    The licence travels with the FILE, so a file whose bytes changed is a different
    work.  This is a one-line guard that stops a replaced texture from keeping the
    original's licence by accident.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    got = _sha256(path)
    want_prefix = GRASS_CC0_ASSET["sha256_prefix"]
    if not got.startswith(want_prefix):
        raise ValueError(
            f"{path} has sha256 {got[:24]}, but the recorded CC0 asset is "
            f"{want_prefix}...: the licence recorded for {GRASS_CC0_ASSET['title']} "
            "belongs to the ORIGINAL file, so this one may not be used under it")


def ensure_grass_texture(size: int = 1024, *, allow_resize: bool = True) -> dict[str, Any]:
    """Resize the CC0 lawn photograph to a square tile PNG and write its provenance.

    The source JPEG is verified against the recorded hash BEFORE any resizing, so the
    chain of custody is: verified source -> derived tile, both recorded.
    """
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    if not SRC_JPG.exists():
        raise FileNotFoundError(
            f"{SRC_JPG} is missing.  Re-fetch it from "
            f"{GRASS_CC0_ASSET['source_page']} and verify its sha256 prefix "
            f"{GRASS_CC0_ASSET['sha256_prefix']}.")
    refuse_unverified_texture(SRC_JPG)
    from PIL import Image
    with Image.open(SRC_JPG) as im:
        im = im.convert("RGB")
        src_size = list(im.size)
        if allow_resize and (im.size[0] != size or im.size[1] != size):
            # CENTRE-CROP then resize, so the tile stays square and seamless.  A plain
            # resize of a 1556x1556 source to 1024x1024 is also square here, but the
            # crop keeps the intent explicit for non-square sources.
            side = min(im.size)
            left = (im.width - side) // 2
            top = (im.height - side) // 2
            im = im.crop((left, top, left + side, top + side)).resize(
                (size, size), Image.LANCZOS)
        im.save(GRASS_TEXTURE_PNG)
    rep = grass_texture_provenance(src_size=src_size, out_size=[int(v) for v in
                                                                Image.open(GRASS_TEXTURE_PNG).size])
    SIDECAR.write_text(json.dumps(rep, indent=2, sort_keys=True))
    return rep


def grass_texture_provenance(src_size: list[int], out_size: list[int]) -> dict[str, Any]:
    return {
        "generator": "ensure_grass_texture",
        "route": GRASS_CC0_ASSET["route"],
        "key": "grass_lawn_cc0",
        "path": str(GRASS_TEXTURE_PNG),
        "source_file_cached_at": str(SRC_JPG),
        "source_page": GRASS_CC0_ASSET["source_page"],
        "source_title": GRASS_CC0_ASSET["title"],
        "source_size_px": src_size,
        "source_sha256_prefix_verified": GRASS_CC0_ASSET["sha256_prefix"],
        "author": GRASS_CC0_ASSET["author"],
        "license_name": GRASS_CC0_ASSET["licence"],
        "license_url": GRASS_CC0_ASSET["licence_url"],
        "date": GRASS_CC0_ASSET["date"],
        "usage_terms": "Creative Commons Zero, Public Domain Dedication",
        "output_size_px": out_size,
        "derivation": ("centre-cropped to a square and resized with LANCZOS; the "
                       "bytes of the source were verified against the recorded "
                       "sha256 before any processing"),
        "what_the_image_is_used_for": GRASS_CC0_ASSET["what"],
    }


def meadow_scene_config(seed: int = 20261004, *, half_extent_mm: float = 200.0,
                        n_samples: int = 801) -> dict[str, Any]:
    """The generation parameters for the meadow, as a recordable dict.

    Kept as a dict rather than a dataclass because it is handed to the terrain
    builder in ``natural_scene`` as kwargs and recorded verbatim in the artefact.
    """
    return {"seed": int(seed), "half_extent_mm": float(half_extent_mm),
            "n_samples": int(n_samples),
            "terrain_kind": "grass",
            "texture": GRASS_CC0_ASSET["file"],
            "preset": MEADOW_PRESET_NAME}


def build_meadow_terrain(seed: int = 20261004, half_extent_mm: float = 200.0,
                         n_samples: int = 801
                         ) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate the meadow heightfield with the parameters the preset will use."""
    return generate_grass_heightfield(n_samples=n_samples, extent_mm=half_extent_mm,
                                      seed=seed)


def meadow_report() -> dict[str, Any]:
    """One dict with the terrain's and the texture's provenance, for the artefact."""
    tex = json.loads(SIDECAR.read_text()) if SIDECAR.exists() else None
    return {"preset": MEADOW_PRESET_NAME, "texture": tex,
            "source_asset": GRASS_CC0_ASSET,
            "texture_png_exists": GRASS_TEXTURE_PNG.exists(),
            "texture_png_bytes": (GRASS_TEXTURE_PNG.stat().st_size
                                  if GRASS_TEXTURE_PNG.exists() else 0)}


if __name__ == "__main__":       # pragma: no cover - manual step
    rep = ensure_grass_texture()
    print(json.dumps({k: rep[k] for k in ("route", "output_size_px", "author",
                                          "license_name")}, indent=2))
    elev, meta = build_meadow_terrain()
    print("terrain:", elev.shape, "envelope_ok", meta["design_envelope_ok"],
          "peak-to-peak %.3f mm" % meta["amplitude_peak_to_peak_mm"])
