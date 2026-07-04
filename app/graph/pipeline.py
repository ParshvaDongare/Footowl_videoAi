from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.agents.nodes import PipelineServices, run_compiler_and_fixer, run_image_analyser, run_renderer, run_script_generator, run_storyboard_writer
from app.models import ArtifactPaths, PipelineState, PipelineStatus
from app.settings import SETTINGS
from app.services.artifact_service import ArtifactManager
from app.services.logging_service import RunLogger
from app.utils import ensure_dir
from app.graph.simple_graph import SimpleStateGraph


def build_graph(services: PipelineServices, artifact_dir: Path, logger: RunLogger) -> SimpleStateGraph:
    graph = SimpleStateGraph()
    graph.add_node("Image Analyser", lambda state: run_image_analyser(state, services, logger))
    graph.add_node("Storyboard Writer", lambda state: run_storyboard_writer(state, services, logger))
    graph.add_node("Script Generator", lambda state: run_script_generator(state, services, logger))
    graph.add_node("Compiler & Fixer", lambda state: run_compiler_and_fixer(state, services, logger))
    graph.add_node("Renderer", lambda state: run_renderer(state, services, logger, artifact_dir))
    graph.add_edge("Image Analyser", "Storyboard Writer")
    graph.add_edge("Storyboard Writer", "Script Generator")

    def choose_next(state: PipelineState) -> str:
        if state.compile_errors and state.retry_count < SETTINGS.retry_limit:
            return "Script Generator"
        return "Renderer" if not state.compile_errors else ""

    graph.add_conditional_edges("Script Generator", lambda state: "Compiler & Fixer")
    graph.add_conditional_edges("Compiler & Fixer", choose_next)
    graph.set_start("Image Analyser")
    return graph


def run_pipeline(
    source_type,
    source_ref: str,
    user_prompt: str,
    images: list,
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
        images=images,
        status=PipelineStatus.running,
        artifact_paths=ArtifactPaths(run_dir=str(artifact_root)),
    )
    state.video_intent = services.ai.parse_intent(user_prompt)
    artifacts.save_json("video_intent.json", state.video_intent.model_dump())
    graph = build_graph(services, artifact_root, logger)
    state = graph.run(state)
    state.status = PipelineStatus.succeeded if not state.compile_errors else PipelineStatus.failed
    state.artifact_paths.storyboard_json = str(artifacts.save_json("storyboard.json", state.storyboard.model_dump()))
    state.artifact_paths.composition_spec_json = str(artifacts.save_json("composition_spec.json", state.composition_spec.model_dump()))
    state.artifact_paths.tsx_script = str(artifacts.save_script("Composition.tsx", state.remotion_code))
    state.artifact_paths.compile_attempts_json = str(artifacts.save_json("compile_attempts.json", state.compile_attempts))
    state.artifact_paths.pipeline_state_json = str(artifacts.save_json("pipeline_state.json", state.model_dump()))
    state.artifact_paths.graph_trace_json = str(artifacts.save_trace(logger.events))
    if state.output_video:
        state.artifact_paths.output_video = str(Path(state.output_video))
    return state
