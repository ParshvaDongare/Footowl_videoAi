from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from app.agents.nodes import (
    PipelineServices,
    run_compiler_and_fixer,
    run_image_analyser,
    run_renderer,
    run_script_generator,
    run_storyboard_writer,
)
from app.models import ArtifactPaths, PipelineState, PipelineStatus
from app.settings import SETTINGS
from app.services.artifact_service import ArtifactManager
from app.services.logging_service import RunLogger
from app.utils import ensure_dir


class GraphState(TypedDict, total=False):
    run_id: str
    current_node: str
    source_type: Any
    source_ref: str
    user_prompt: str
    video_intent: dict[str, Any]
    images: list[dict[str, Any]]
    image_analysis: list[dict[str, Any]]
    selected_images: list[dict[str, Any]]
    selected_event_cluster: str
    event_clusters: list[dict[str, Any]]
    retrieval_context: dict[str, Any]
    repair_context: dict[str, Any]
    storyboard: dict[str, Any]
    composition_spec: dict[str, Any]
    validation_result: dict[str, Any]
    remotion_code: str
    compile_errors: list[dict[str, Any]]
    compile_attempts: list[dict[str, Any]]
    retry_count: int
    status: str
    output_video: str | None
    artifact_paths: dict[str, Any] | None
    created_at: str


def build_graph(services: PipelineServices, artifact_dir: Path, logger: RunLogger):
    graph: StateGraph[GraphState] = StateGraph(GraphState)

    def wrap(node_name: str, handler, extra: dict[str, Any] | None = None):
        def _node(state: GraphState) -> GraphState:
            model = PipelineState.model_validate(state)
            model.current_node = node_name
            updated = handler(model, services, logger, *(extra or {}).get("args", []))
            updated.current_node = node_name
            return updated.model_dump()

        return _node

    graph.add_node("Image Analyser", wrap("Image Analyser", run_image_analyser))
    graph.add_node("Storyboard Writer", wrap("Storyboard Writer", run_storyboard_writer))
    graph.add_node("Script Generator", wrap("Script Generator", run_script_generator))
    graph.add_node("Compiler & Fixer", wrap("Compiler & Fixer", run_compiler_and_fixer))
    graph.add_node("Renderer", wrap("Renderer", run_renderer, {"args": [artifact_dir]}))

    graph.add_edge(START, "Image Analyser")
    graph.add_edge("Image Analyser", "Storyboard Writer")
    graph.add_edge("Storyboard Writer", "Script Generator")
    graph.add_edge("Script Generator", "Compiler & Fixer")

    def route_after_compiler(state: GraphState) -> str:
        model = PipelineState.model_validate(state)
        if model.compile_errors and model.retry_count < SETTINGS.retry_limit:
            return "retry"
        if model.compile_errors:
            return "end"
        return "render"

    graph.add_conditional_edges(
        "Compiler & Fixer",
        route_after_compiler,
        {
            "retry": "Script Generator",
            "render": "Renderer",
            "end": END,
        },
    )
    graph.add_edge("Renderer", END)
    return graph.compile()


def run_pipeline(
    source_type,
    source_ref: str,
    user_prompt: str,
    images: list,
    video_width: int | None = None,
    video_height: int | None = None,
    video_fps: int | None = None,
    target_duration_seconds: int | None = None,
    output_root: Path | None = None,
    services: PipelineServices | None = None,
) -> PipelineState:
    run_id = uuid4().hex[:10]
    artifact_root = ensure_dir((output_root or SETTINGS.output_root) / run_id)
    artifacts = ArtifactManager(artifact_root)
    logger = RunLogger(run_id=run_id)
    services = services or PipelineServices()
    state = PipelineState(
        run_id=run_id,
        source_type=source_type,
        source_ref=source_ref,
        user_prompt=user_prompt,
        video_width=video_width or SETTINGS.default_width,
        video_height=video_height or SETTINGS.default_height,
        video_fps=video_fps or SETTINGS.default_fps,
        images=images,
        status=PipelineStatus.running,
        artifact_paths=ArtifactPaths(run_dir=str(artifact_root)),
    )
    logger.log("Input Adapter", "start", source_type=str(source_type), source_ref=source_ref)
    state.video_intent = services.ai.parse_intent(user_prompt)
    if target_duration_seconds:
        state.video_intent.target_duration_seconds = target_duration_seconds
    logger.log("Input Adapter", "end", video_intent=state.video_intent.model_dump())
    artifacts.save_json("video_intent.json", state.video_intent.model_dump())

    graph = build_graph(services, artifact_root, logger)
    final_state = PipelineState.model_validate(graph.invoke(state.model_dump()))
    final_state.status = PipelineStatus.succeeded if final_state.output_video and not final_state.compile_errors else PipelineStatus.failed

    final_state.artifact_paths.storyboard_json = str(artifact_root / "storyboard.json")
    final_state.artifact_paths.composition_spec_json = str(artifact_root / "composition_spec.json")
    final_state.artifact_paths.tsx_script = str(artifact_root / "Composition.tsx")
    final_state.artifact_paths.compile_attempts_json = str(artifact_root / "compile_attempts.json")
    final_state.artifact_paths.pipeline_state_json = str(artifact_root / "pipeline_state.json")
    final_state.artifact_paths.graph_trace_json = str(artifact_root / SETTINGS.graph_trace_name)
    if final_state.output_video:
        final_state.artifact_paths.output_video = str(Path(final_state.output_video))

    artifacts.save_json("storyboard.json", final_state.storyboard.model_dump() if final_state.storyboard else {})
    artifacts.save_json("composition_spec.json", final_state.composition_spec.model_dump() if final_state.composition_spec else {})
    artifacts.save_script("Composition.tsx", final_state.remotion_code)
    artifacts.save_json("compile_attempts.json", final_state.compile_attempts)
    artifacts.save_json("pipeline_state.json", final_state.model_dump())
    artifacts.save_json(SETTINGS.graph_trace_name, logger.events)
    return final_state
