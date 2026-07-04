from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class AppSettings:
    model_intent: str = os.getenv("MODEL_INTENT", "heuristic-intent")
    model_vision: str = os.getenv("MODEL_VISION", "heuristic-vision")
    model_storyboard: str = os.getenv("MODEL_STORYBOARD", "heuristic-storyboard")
    model_script: str = os.getenv("MODEL_SCRIPT", "heuristic-script")
    model_fix: str = os.getenv("MODEL_FIX", "heuristic-fix")
    model_judge: str = os.getenv("MODEL_JUDGE", "heuristic-judge")
    retry_limit: int = int(os.getenv("RETRY_LIMIT", "3"))
    top_k_retrieval: int = int(os.getenv("TOP_K_RETRIEVAL", "3"))
    max_selected_images: int = int(os.getenv("MAX_SELECTED_IMAGES", "6"))
    max_storyboard_scenes: int = int(os.getenv("MAX_STORYBOARD_SCENES", "6"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    output_root: Path = Path(os.getenv("OUTPUT_ROOT", "output"))
    graph_trace_name: str = os.getenv("GRAPH_TRACE_NAME", "graph_trace.json")
    default_width: int = int(os.getenv("VIDEO_WIDTH", "1280"))
    default_height: int = int(os.getenv("VIDEO_HEIGHT", "720"))
    default_fps: int = int(os.getenv("VIDEO_FPS", "24"))


SETTINGS = AppSettings()

