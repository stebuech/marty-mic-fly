"""Dotted-key Patches auf Pipeline-YAML-dicts.

Unterstützte Pfad-Syntax:
  - "a.b.c"            normale dict-Verschachtelung
  - "a[0].b"           Liste mit numerischem Index
  - "array_filter.x.y" Convenience: erster Stage mit kind=array_filter

Mutationsfrei: gibt eine deep-copy zurück."""
from __future__ import annotations

import copy
import re
from typing import Any


_INDEX_RE = re.compile(r"^([^[]+)\[(\d+)\]$")


def _find_array_filter_index(stages: list) -> int:
    for i, s in enumerate(stages):
        if isinstance(s, dict) and s.get("kind") == "array_filter":
            return i
    raise KeyError("no stage with kind='array_filter' found")


def _navigate(node: Any, parts: list[str]) -> tuple[Any, str]:
    """Walk parts[:-1], return (parent, last_key)."""
    for p in parts[:-1]:
        m = _INDEX_RE.match(p)
        if m:
            key, idx = m.group(1), int(m.group(2))
            node = node[key][idx]
        else:
            node = node[p]
    return node, parts[-1]


def apply_overrides(base: dict, overrides: dict[str, Any]) -> dict:
    """Apply each (dotted_key, value) override to a deep-copy of base."""
    out = copy.deepcopy(base)
    for path, value in overrides.items():
        if path.startswith("array_filter."):
            af_idx = _find_array_filter_index(out["stages"])
            real_path = f"stages[{af_idx}]." + path[len("array_filter."):]
        else:
            real_path = path
        parts = real_path.split(".")
        try:
            parent, last = _navigate(out, parts)
            m = _INDEX_RE.match(last)
            if m:
                key, idx = m.group(1), int(m.group(2))
                parent[key][idx] = value
            else:
                if isinstance(parent, dict) and last not in parent:
                    raise KeyError(f"path not found: {path}")
                parent[last] = value
        except (KeyError, TypeError, IndexError) as exc:
            raise KeyError(f"failed to apply override {path}: {exc}") from exc
    return out
