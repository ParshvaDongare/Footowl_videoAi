from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class AppSettings:
    model_intent: str = os.getenv("MODEL_INTENT", os.getenv("GEMINI_MODEL_INTENT", "gemini-3.1-flash-lite"))
    model_vision: str = os.getenv("MODEL_VISION", os.getenv("GEMINI_MODEL_VISION", "gemini-3.5-flash"))
    model_storyboard: str = os.getenv("MODEL_STORYBOARD", os.getenv("GEMINI_MODEL_STORYBOARD", "gemini-3.1-flash-lite"))
    model_script: str = os.getenv("MODEL_SCRIPT", os.getenv("GEMINI_MODEL_SCRIPT", "gemini-3.5-flash"))
    model_fix: str = os.getenv("MODEL_FIX", os.getenv("GEMINI_MODEL_FIX", "gemini-3.1-flash-lite"))
    model_judge: str = os.getenv("MODEL_JUDGE", os.getenv("GEMINI_MODEL_JUDGE", "gemini-3.1-flash-lite"))
    retry_limit: int = int(os.getenv("RETRY_LIMIT", "3"))
    top_k_retrieval: int = int(os.getenv("TOP_K_RETRIEVAL", "3"))
    max_selected_images: int = int(os.getenv("MAX_SELECTED_IMAGES", "6"))
    max_storyboard_scenes: int = int(os.getenv("MAX_STORYBOARD_SCENES", "6"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    chroma_path: Path = Path(os.getenv("CHROMA_PATH", "chroma_store_runtime"))
    llm_cache_path: Path = Path(os.getenv("LLM_CACHE_PATH", "llm_cache"))
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini")
    gemini_call_budget: int = int(os.getenv("GEMINI_CALL_BUDGET", "6"))
    output_root: Path = Path(os.getenv("OUTPUT_ROOT", "output"))
    graph_trace_name: str = os.getenv("GRAPH_TRACE_NAME", "graph_trace.json")
    default_width: int = int(os.getenv("VIDEO_WIDTH", "1280"))
    default_height: int = int(os.getenv("VIDEO_HEIGHT", "720"))
    default_fps: int = int(os.getenv("VIDEO_FPS", "24"))


SETTINGS = AppSettings()
