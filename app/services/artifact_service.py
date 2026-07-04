from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from app.utils import ensure_dir


class ArtifactManager:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = ensure_dir(run_dir)

    def save_json(self, name: str, payload: Any) -> Path:
        path = self.run_dir / name
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def save_script(self, name: str, content: str) -> Path:
        path = self.run_dir / name
        path.write_text(content, encoding="utf-8")
        return path

    def save_text(self, name: str, content: str) -> Path:
        return self.save_script(name, content)

    def save_trace(self, payload: Any) -> Path:
        return self.save_json("graph_trace.json", payload)

    def save_video(self, path: Path, name: str = "output.mp4") -> Path:
        target = self.run_dir / name
        target.write_bytes(path.read_bytes())
        return target

