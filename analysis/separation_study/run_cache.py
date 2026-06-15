"""Run-Caching: stabiler Hash über config-dict + Detection ob Run schon fertig."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def config_hash(cfg: Any) -> str:
    """Deterministischer 12-stelliger SHA1 über sortiertes JSON."""
    serialized = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:12]


def is_run_complete(run_dir: Path) -> bool:
    """True wenn metrics.json und study_metrics.json beide existieren —
    entweder direkt in run_dir oder in einem Subdirectory (Pipeline nestet
    Outputs unter <output_dir>/<auto_run_id>/)."""
    if not run_dir.exists():
        return False
    if ((run_dir / "metrics.json").is_file()
            and (run_dir / "study_metrics.json").is_file()):
        return True
    for sub in run_dir.iterdir():
        if (sub.is_dir()
                and (sub / "metrics.json").is_file()
                and (sub / "study_metrics.json").is_file()):
            return True
    return False
