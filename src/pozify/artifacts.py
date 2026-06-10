from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pozify.contracts import to_dict


def write_json(run_dir: Path, filename: str, payload: Any) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / filename
    path.write_text(json.dumps(to_dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return path

