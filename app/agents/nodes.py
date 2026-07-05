from __future__ import annotations

from pathlib import Path
from typing import Any
from time import perf_counter

from app.models import PipelineState, PipelineStatus
from app.services.ai_service import AIService
from app.services.compiler_service import CompilerService
from app.services.input_adapter import InputAdapter
from app.services.rendering_service import RenderingService
from app.services.retrieval_service import RetrievalService
from app.services.vision_service import VisionService
from app.settings import SETTINGS
from app.utils import ensure_dir


class PipelineServices:
    def __init__(self) -> None:
        self.input_adapter = InputAdapter()
        self.vision = VisionService()
        self.ai = AIService()
        self.retrieval = RetrievalService()
        self.compiler = CompilerService()
        self.renderer = RenderingService()


def run_image_analyser(state: PipelineState, services: PipelineServices, logger) -> PipelineState:
    started = perf_counter()
    logger.log(state.current_node, "start", node_input={"prompt": state.user_prompt, "image_count": len(state.images)})
    analyses = services.vision.analyze_batch(state.images)
    selected, cluster_info = services.ai.select_images(analyses, state.video_intent, SETTINGS.max_selected_images)
    state.image_analysis = analyses
    state.selected_images = selected
    state.selected_event_cluster = cluster_info["selected_event_cluster"]
    state.event_clusters = cluster_info["event_clusters"]
    logger.log(
        state.current_node,
        "end",
        node_output={
            "analysed": len(analyses),
            "selected": [img.image_id for img in selected],
            "cluster": state.selected_event_cluster,
            "latency_ms": round((perf_counter() - started) * 1000.0, 2),
        },
    )
    return state


def run_storyboard_writer(state: PipelineState, services: PipelineServices, logger) -> PipelineState:
    started = perf_counter()
    logger.log(state.current_node, "start", node_input={"intent": state.video_intent.model_dump()})
    style_hits = services.retrieval.retrieve_style(state.video_intent.visual_style, SETTINGS.top_k_retrieval)
    storyboard = services.ai.write_storyboard(state.video_intent, style_hits, state.selected_images)
    if state.selected_event_cluster:
        storyboard.notes.insert(0, f"Selected event cluster: {state.selected_event_cluster}")
    spec = services.ai.plan_composition(storyboard, state.video_width, state.video_height, state.video_fps)
    state.retrieval_context["style_guides"] = style_hits
    state.retrieval_context["selected_event_cluster"] = state.selected_event_cluster
    state.storyboard = storyboard
    state.composition_spec = spec
    logger.log(
        state.current_node,
        "end",
        node_output={
            "scenes": len(storyboard.scenes),
            "duration": spec.duration_seconds,
            "cluster": state.selected_event_cluster,
            "latency_ms": round((perf_counter() - started) * 1000.0, 2),
        },
    )
    return state


def run_script_generator(state: PipelineState, services: PipelineServices, logger) -> PipelineState:
    started = perf_counter()
    logger.log(state.current_node, "start", node_input={"scenes": len(state.composition_spec.scenes)})
    validation = services.ai.validate_spec(state.composition_spec, state.selected_images)
    state.validation_result = validation
    if not validation.is_valid:
        raise ValueError("CompositionSpec validation failed: " + "; ".join(validation.errors))
    remotion_hits = services.retrieval.retrieve_remotion(state.composition_spec.title, SETTINGS.top_k_retrieval)
    state.retrieval_context["remotion"] = remotion_hits
    asset_lookup = {img.image_id: img.path for img in state.images}
    state.remotion_code = services.ai.generate_tsx(state.composition_spec, asset_lookup, repair_context=state.repair_context)
    logger.log(
        state.current_node,
        "end",
        node_output={
            "tsx_length": len(state.remotion_code),
            "retrieved": len(remotion_hits),
            "repair_context": bool(state.repair_context),
            "latency_ms": round((perf_counter() - started) * 1000.0, 2),
        },
    )
    return state


def run_compiler_and_fixer(state: PipelineState, services: PipelineServices, logger) -> PipelineState:
    started = perf_counter()
    logger.log(state.current_node, "start", node_input={"retry_count": state.retry_count})
    asset_lookup = {img.image_id: img.path for img in state.images}
    attempt = services.compiler.compile(state.remotion_code, state.composition_spec, list(asset_lookup.values()))
    state.compile_attempts.append(
        {
            "attempt": state.retry_count + 1,
            "ok": attempt.ok,
            "diagnostics": [d.model_dump() for d in attempt.diagnostics],
        }
    )
    state.compile_errors = attempt.diagnostics
    if attempt.ok:
        state.repair_context = {}
        logger.log(
            state.current_node,
            "end",
            node_output={"ok": True, "latency_ms": round((perf_counter() - started) * 1000.0, 2)},
        )
        return state

    state.retry_count += 1
    error_class = services.compiler.classify(attempt.diagnostics)
    fix_docs = services.retrieval.retrieve_remotion(error_class, SETTINGS.top_k_retrieval)
    state.repair_context = {
        "retry_count": state.retry_count,
        "error_class": error_class,
        "diagnostics": [d.model_dump() for d in attempt.diagnostics],
        "retrieved_docs": fix_docs,
        "asset_lookup": asset_lookup,
    }
    state.retrieval_context["compiler_errors"] = [d.model_dump() for d in attempt.diagnostics]
    state.retrieval_context["compiler_fix_docs"] = fix_docs
    state.storyboard = services.ai.repair_storyboard_for_error(state.storyboard, attempt.diagnostics)
    state.composition_spec = services.ai.plan_composition(state.storyboard, state.composition_spec.width, state.composition_spec.height, state.composition_spec.fps)
    logger.log(
        state.current_node,
        "end",
        node_output={
            "ok": False,
            "retry_count": state.retry_count,
            "error_class": error_class,
            "diagnostics": [d.model_dump() for d in attempt.diagnostics],
            "latency_ms": round((perf_counter() - started) * 1000.0, 2),
        },
    )
    return state


def run_renderer(state: PipelineState, services: PipelineServices, logger, artifact_dir: Path) -> PipelineState:
    started = perf_counter()
    logger.log(state.current_node, "start", node_input={"scenes": len(state.composition_spec.scenes)})
    asset_lookup = {img.image_id: img.path for img in state.images}
    output = artifact_dir / "output.mp4"
    services.renderer.render(state.composition_spec, output, asset_lookup, intent=state.video_intent)
    state.output_video = str(output)
    logger.log(
        state.current_node,
        "end",
        node_output={"output_video": state.output_video, "latency_ms": round((perf_counter() - started) * 1000.0, 2)},
    )
    return state
