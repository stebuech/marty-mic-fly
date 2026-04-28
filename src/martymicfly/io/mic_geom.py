"""Load Acoular-format MicArray XML."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def load_mic_geom_xml(path: str | Path) -> np.ndarray:
    """Parse `<MicArray>`/`<pos>` XML, return positions as ``(M, 3)`` float64.

    Reihenfolge entspricht dem Auftreten der ``<pos>``-Elemente.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    tree = ET.parse(path)
    root = tree.getroot()
    rows: list[tuple[float, float, float]] = []
    for el in root.findall("pos"):
        try:
            x = float(el.attrib["x"])
            y = float(el.attrib["y"])
            z = float(el.attrib["z"])
        except KeyError as exc:
            missing = exc.args[0]
            raise ValueError(
                f"{path}: <pos> missing attribute {missing!r}"
            ) from exc
        rows.append((x, y, z))
    if not rows:
        raise ValueError(f"{path}: no <pos> elements found")
    return np.array(rows, dtype=np.float64)
