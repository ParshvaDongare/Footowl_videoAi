from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, Field


class SourceType(str, Enum):
    local_folder = "local_folder"
    google_drive_folder = "google_drive_folder"


class PipelineStatus(str, Enum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class VideoIntent(BaseModel):
    pacing: Literal["slow", "moderate", "fast"]
    visual_style: str
    caption_tone: str
    transition_preference: str
    color_treatment: str
    target_duration_seconds: int
    must_include: list[str] = Field(default_factory=list)
    must_avoid: list[str] = Field(default_factory=list)


class ImageRecord(BaseModel):
    image_id: str
    path: str
    width: int
    height: int
    file_size: int


class ImageAnalysis(BaseModel):
    image_id: str
    image_summary: str
    perceptual_hash: str = ""
    palette_family: str = ""
    people: int
    objects: list[str] = Field(default_factory=list)
    event_type: str = "general"  # e.g. wedding, party, performance, sport, general
    emotion: str
    blur_score: float
    quality_score: float
    indoor_outdoor: str
    aesthetic_score: float
    duplicate_score: float
    confidence: float
    rank_score: float


class ScenePlan(BaseModel):
    scene_id: str
    image_id: str
    duration_seconds: float
    caption: str
    transition: str
    animation: str
    timing: str
    rationale: str


class Storyboard(BaseModel):
    title: str
    logline: str
    tone: str
    scenes: list[ScenePlan] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CompositionScene(BaseModel):
    scene_id: str
    image_id: str
    duration_seconds: float
    caption: str
    transition: str
    animation: str
    timing: str
    start_seconds: float


class CompositionSpec(BaseModel):
    title: str
    width: int
    height: int
    fps: int
    duration_seconds: float
    scenes: list[CompositionScene] = Field(default_factory=list)
    render_notes: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CompileDiagnostic(BaseModel):
    category: str
    message: str
    severity: Literal["error", "warning"] = "error"
    file: str | None = None
    line: int | None = None


class ArtifactPaths(BaseModel):
    run_dir: str
    storyboard_json: str | None = None
    composition_spec_json: str | None = None
    tsx_script: str | None = None
    compile_attempts_json: str | None = None
    pipeline_state_json: str | None = None
    graph_trace_json: str | None = None
    output_video: str | None = None


class PipelineState(BaseModel):
    run_id: str
    current_node: str = "input"
    source_type: SourceType
    source_ref: str
    user_prompt: str
    video_intent: VideoIntent | None = None
    images: list[ImageRecord] = Field(default_factory=list)
    image_analysis: list[ImageAnalysis] = Field(default_factory=list)
    selected_images: list[ImageAnalysis] = Field(default_factory=list)
    selected_event_cluster: str = ""
    event_clusters: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_context: dict[str, Any] = Field(default_factory=dict)
    storyboard: Storyboard | None = None
    composition_spec: CompositionSpec | None = None
    validation_result: ValidationResult | None = None
    remotion_code: str = ""
    compile_errors: list[CompileDiagnostic] = Field(default_factory=list)
    compile_attempts: list[dict[str, Any]] = Field(default_factory=list)
    retry_count: int = 0
    status: PipelineStatus = PipelineStatus.pending
    output_video: str | None = None
    artifact_paths: ArtifactPaths | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
