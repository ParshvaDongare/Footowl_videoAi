from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from app.models import (
    CompileDiagnostic,
    CompositionScene,
    CompositionSpec,
    JudgeAssessment,
    ImageAnalysis,
    ScriptGenerationResult,
    ScenePlan,
    Storyboard,
    ValidationResult,
    VideoIntent,
)
from app.settings import SETTINGS
from app.services.llm_service import LLMService
from app.services.prompt_loader import load_prompt
from app.utils import clamp


@dataclass
class AIService:
    llm: LLMService = field(default_factory=LLMService)

    def __post_init__(self) -> None:
        self.prompt_intent: str = load_prompt("intent")
        self.prompt_storyboard: str = load_prompt("storyboard")
        self.prompt_composition: str = load_prompt("composition")
        self.prompt_fix: str = load_prompt("fix")
        self.prompt_script: str = load_prompt("script")
        self.prompt_judge: str = load_prompt("judge")

    def parse_intent(self, prompt: str) -> VideoIntent:
        if self.llm.available:
            structured = self.llm.generate_json(
                model=SETTINGS.model_intent,
                system_prompt=self.prompt_intent,
                user_prompt=(
                    "Convert the following creative brief into a strict VideoIntent JSON object.\n"
                    "Return JSON only.\n\n"
                    f"Brief:\n{prompt}"
                ),
                response_model=VideoIntent,
                temperature=SETTINGS.temperature,
            )
            if structured:
                return structured
        return self._heuristic_intent(prompt)

    def select_images(
        self,
        analyses: list[ImageAnalysis],
        intent: VideoIntent,
        max_selected_images: int,
    ) -> tuple[list[ImageAnalysis], dict[str, Any]]:
        clusters = self._cluster_images(analyses)
        if not clusters:
            return [], {"selected_event_cluster": "", "event_clusters": []}

        best_cluster = max(clusters, key=lambda c: (c["total_rank"], c["size"], c["avg_similarity"]))
        ordered = sorted(best_cluster["items"], key=lambda a: (a.rank_score, a.confidence, a.quality_score), reverse=True)
        representative = best_cluster["representative"]
        coherent_pool = [
            item
            for item in ordered
            if self._event_similarity(item, representative) >= 0.72
            and (
                item.palette_family == representative.palette_family
                or item.event_type == representative.event_type
            )
        ]
        if coherent_pool:
            ordered = coherent_pool

        selected: list[ImageAnalysis] = []
        for item in ordered:
            if self._too_similar(item, selected):
                continue
            selected.append(item)
            if len(selected) >= max_selected_images:
                break

        if not selected:
            selected = ordered[: min(max_selected_images, len(ordered))]

        cluster_info = {
            "selected_event_cluster": best_cluster["label"],
            "event_clusters": [
                {
                    "label": cluster["label"],
                    "size": cluster["size"],
                    "total_rank": cluster["total_rank"],
                    "avg_similarity": cluster["avg_similarity"],
                    "event_types": cluster["event_types"],
                    "palette_families": cluster["palette_families"],
                    "image_ids": [item.image_id for item in cluster["items"]],
                }
                for cluster in clusters
            ],
        }
        return selected, cluster_info

    def write_storyboard(
        self,
        intent: VideoIntent,
        style_context: list[dict[str, Any]],
        selected_images: list[ImageAnalysis],
    ) -> Storyboard:
        style_note = style_context[0]["text"] if style_context else intent.visual_style
        capped_images = selected_images[: SETTINGS.max_storyboard_scenes]
        if self.llm.available:
            payload = {
                "intent": intent.model_dump(),
                "style_context": style_context,
                "selected_images": [img.model_dump() for img in capped_images],
                "max_storyboard_scenes": SETTINGS.max_storyboard_scenes,
            }
            structured = self.llm.generate_json(
                model=SETTINGS.model_storyboard,
                system_prompt=self.prompt_storyboard,
                user_prompt=(
                    "Write a storyboard JSON object for a reel.\n"
                    "Use the selected images only. Preserve the intent and style tone.\n"
                    "Return valid JSON matching the Storyboard schema.\n\n"
                    f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
                ),
                response_model=Storyboard,
                temperature=SETTINGS.temperature,
            )
            if structured and structured.scenes:
                return self._normalize_storyboard(structured, style_note, capped_images)

        intro = f"{intent.visual_style.title()} reel with {intent.caption_tone} captions."
        scenes: list[ScenePlan] = []
        total = max(intent.target_duration_seconds, len(capped_images) * 3)
        base = total / max(len(capped_images), 1)
        for index, image in enumerate(capped_images, start=1):
            duration = clamp(base * (1.2 if index == 1 else 1.0), 2.0, 6.0)
            scenes.append(
                ScenePlan(
                    scene_id=f"scene_{index}",
                    image_id=image.image_id,
                    duration_seconds=duration,
                    caption=self._caption_for(intent, image, index),
                    transition=self._transition_for(intent, index),
                    animation=self._animation_for(intent, index),
                    timing="opening" if index == 1 else "mid" if index < len(capped_images) else "ending",
                    rationale=f"Selected for {image.image_summary} and {style_note}",
                )
            )
        return Storyboard(
            title=f"{intent.visual_style.title()} Highlight Reel",
            logline=intro,
            tone=intent.caption_tone,
            scenes=scenes[:],
            notes=[style_note, f"Selected {len(scenes)} scenes from {len(capped_images)} candidate images."],
        )

    def plan_composition(
        self,
        storyboard: Storyboard,
        width: int,
        height: int,
        fps: int,
    ) -> CompositionSpec:
        if self.llm.available:
            structured = self.llm.generate_json(
                model=SETTINGS.model_storyboard,
                system_prompt=self.prompt_composition,
                user_prompt=(
                    "Convert the following storyboard into a CompositionSpec JSON object.\n"
                    "Keep the same scenes and ensure start_seconds are sequential.\n"
                    "Return valid JSON matching the CompositionSpec schema.\n\n"
                    f"Canvas: {width}x{height} @ {fps} fps\n\n"
                    f"{storyboard.model_dump_json(indent=2)}"
                ),
                response_model=CompositionSpec,
                temperature=SETTINGS.temperature,
            )
            if structured and structured.scenes:
                return self._normalize_composition_spec(structured, width, height, fps)

        scenes: list[CompositionScene] = []
        start = 0.0
        for scene in storyboard.scenes[: SETTINGS.max_storyboard_scenes]:
            scenes.append(
                CompositionScene(
                    scene_id=scene.scene_id,
                    image_id=scene.image_id,
                    duration_seconds=scene.duration_seconds,
                    caption=scene.caption,
                    transition=scene.transition,
                    animation=scene.animation,
                    timing=scene.timing,
                    start_seconds=start,
                )
            )
            start += scene.duration_seconds
        return CompositionSpec(
            title=storyboard.title,
            width=width,
            height=height,
            fps=fps,
            duration_seconds=start,
            scenes=scenes,
            render_notes=["Generated from validated CompositionSpec."],
        )

    def validate_spec(self, spec: CompositionSpec, selected_images: list[ImageAnalysis]) -> ValidationResult:
        errors: list[str] = []
        image_ids = {img.image_id for img in selected_images}
        if not spec.scenes:
            errors.append("CompositionSpec must contain at least one scene.")
        if spec.width <= 0 or spec.height <= 0:
            errors.append("Canvas dimensions must be positive.")
        if spec.fps <= 0:
            errors.append("FPS must be positive.")
        for scene in spec.scenes:
            if scene.image_id not in image_ids:
                errors.append(f"Unknown image_id referenced in scene {scene.scene_id}: {scene.image_id}")
            if scene.duration_seconds <= 0:
                errors.append(f"Scene {scene.scene_id} has non-positive duration.")
            if scene.start_seconds < 0:
                errors.append(f"Scene {scene.scene_id} has invalid start time.")
        return ValidationResult(is_valid=not errors, errors=errors)

    def generate_tsx(
        self,
        spec: CompositionSpec,
        asset_lookup: dict[str, str] | None = None,
        repair_context: dict[str, Any] | None = None,
    ) -> str:
        repair_context = repair_context or {}
        model_name = SETTINGS.model_fix if repair_context else SETTINGS.model_script
        prompt = self.prompt_fix if repair_context else self.prompt_script
        if self.llm.available:
            structured = self.llm.generate_json(
                model=model_name,
                system_prompt=prompt,
                user_prompt=(
                    "Generate a runnable Remotion Composition.tsx file as JSON with a single key `tsx_code`.\n"
                    "You must preserve the provided scenes and asset paths.\n"
                    "Use the repair context to fix any compiler diagnostics or missing assets.\n"
                    "Return valid JSON matching ScriptGenerationResult.\n\n"
                    f"Repair context:\n{json.dumps(repair_context, ensure_ascii=False, indent=2)}\n\n"
                    f"CompositionSpec:\n{spec.model_dump_json(indent=2)}\n\n"
                    f"Asset lookup:\n{json.dumps(asset_lookup or {}, ensure_ascii=False, indent=2)}"
                ),
                response_model=ScriptGenerationResult,
                temperature=SETTINGS.temperature,
            )
            if structured and structured.tsx_code.strip():
                return structured.tsx_code

        return self._fallback_tsx(spec, asset_lookup or {})

    def repair_storyboard_for_error(
        self,
        storyboard: Storyboard,
        diagnostics: list[CompileDiagnostic],
    ) -> Storyboard:
        categories = {d.category for d in diagnostics}
        if "asset" in categories:
            missing_ids: set[str] = set()
            for d in diagnostics:
                if d.category == "asset":
                    parts = d.message.split(":")
                    if len(parts) >= 2:
                        missing_ids.add(parts[-1].strip())
            if missing_ids and len(storyboard.scenes) > len(missing_ids):
                storyboard.scenes = [scene for scene in storyboard.scenes if scene.image_id not in missing_ids]
        return storyboard

    def judge_storyboard(self, storyboard: Storyboard, prompt: str) -> dict[str, Any]:
        if self.llm.available:
            structured = self.llm.generate_json(
                model=SETTINGS.model_judge,
                system_prompt=self.prompt_judge,
                user_prompt=(
                    "Judge the narrative coherence of the storyboard against the prompt.\n"
                    "Return valid JSON matching JudgeAssessment with a coherence_score from 0 to 1.\n\n"
                    f"Prompt:\n{prompt}\n\nStoryboard:\n{storyboard.model_dump_json(indent=2)}"
                ),
                response_model=JudgeAssessment,
                temperature=0.0,
            )
            if structured:
                return structured.model_dump()

        score = 0.6
        keywords = set(prompt.lower().split())
        if any(word in storyboard.logline.lower() for word in keywords):
            score += 0.2
        if len(storyboard.scenes) >= 2:
            score += 0.1
        if storyboard.scenes and storyboard.scenes[0].timing == "opening":
            score += 0.1
        return {
            "coherence_score": min(score, 1.0),
            "notes": ["Storyboard matches prompt intent and has an opening/mid/end arc."],
        }

    def _normalize_storyboard(
        self,
        storyboard: Storyboard,
        style_note: str,
        selected_images: list[ImageAnalysis],
    ) -> Storyboard:
        if not storyboard.scenes:
            return storyboard
        storyboard.scenes = storyboard.scenes[: SETTINGS.max_storyboard_scenes]
        if not storyboard.title:
            storyboard.title = "Highlight Reel"
        if not storyboard.logline:
            storyboard.logline = style_note
        if not storyboard.notes:
            storyboard.notes = [style_note, f"Selected {len(selected_images)} candidate images."]
        return storyboard

    def _normalize_composition_spec(
        self,
        spec: CompositionSpec,
        width: int,
        height: int,
        fps: int,
    ) -> CompositionSpec:
        start = 0.0
        scenes: list[CompositionScene] = []
        for scene in spec.scenes[: SETTINGS.max_storyboard_scenes]:
            scenes.append(
                CompositionScene(
                    scene_id=scene.scene_id,
                    image_id=scene.image_id,
                    duration_seconds=max(scene.duration_seconds, 0.25),
                    caption=scene.caption,
                    transition=scene.transition,
                    animation=scene.animation,
                    timing=scene.timing,
                    start_seconds=start,
                )
            )
            start += max(scene.duration_seconds, 0.25)
        spec.scenes = scenes
        spec.width = width
        spec.height = height
        spec.fps = fps
        spec.duration_seconds = start
        if not spec.render_notes:
            spec.render_notes = ["Generated from validated CompositionSpec."]
        return spec

    def _fallback_tsx(self, spec: CompositionSpec, asset_lookup: dict[str, str]) -> str:
        lines = [
            "import React from 'react';",
            "import {AbsoluteFill, Img, useCurrentFrame, useVideoConfig} from 'remotion';",
            "",
            "const assetPaths: Record<string, string> = {",
        ]
        for image_id, path in asset_lookup.items():
            safe_path = path.replace("\\", "/")
            lines.append(f"  {image_id!r}: {safe_path!r},")
        lines.extend(
            [
                "};",
                "",
                "const scenes = [",
            ]
        )
        for scene in spec.scenes:
            lines.append(
                "  {"
                f" id: {scene.scene_id!r}, imageId: {scene.image_id!r}, start: {scene.start_seconds:.2f},"
                f" duration: {scene.duration_seconds:.2f}, caption: {scene.caption!r},"
                f" transition: {scene.transition!r}, animation: {scene.animation!r} "
                "},"
            )
        lines.extend(
            [
                "];",
                "",
                "export const Composition: React.FC = () => {",
                "  const frame = useCurrentFrame();",
                "  const {fps, width, height} = useVideoConfig();",
                "  const active = scenes.find((scene) => frame >= scene.start * fps && frame < (scene.start + scene.duration) * fps) ?? scenes[0];",
                "  const src = active ? (assetPaths[active.imageId] ?? '') : '';",
                "  return (",
                "    <AbsoluteFill style={{backgroundColor: '#0b0d10', justifyContent: 'center', alignItems: 'center'}}>",
                "      <AbsoluteFill style={{opacity: 0.92}}>",
                "        {active && src && (",
                "          <Img src={src} style={{width: '100%', height: '100%', objectFit: 'cover'}} />",
                "        )}",
                "      </AbsoluteFill>",
                "      <AbsoluteFill style={{justifyContent: 'flex-end', padding: 56, color: 'white'}}>",
                "        <div style={{fontSize: 40, fontWeight: 700, marginBottom: 12}}>{active?.caption}</div>",
                "        <div style={{fontSize: 20, opacity: 0.85}}>{`fps ${fps} · ${width}x${height}`}</div>",
                "      </AbsoluteFill>",
                "    </AbsoluteFill>",
                "  );",
                "};",
            ]
        )
        return "\n".join(lines) + "\n"

    def _heuristic_intent(self, prompt: str) -> VideoIntent:
        text = prompt.lower()
        if any(k in text for k in ("fast", "energetic", "upbeat", "bold")):
            pacing = "fast"
        elif any(k in text for k in ("slow", "emotional", "cinematic", "warm")):
            pacing = "slow"
        else:
            pacing = "moderate"
        if "corporate" in text or "professional" in text:
            style = "clean corporate"
            tone = "concise and polished"
            color = "neutral"
            transition = "subtle fades"
        elif "birthday" in text or "party" in text:
            style = "upbeat celebration"
            tone = "bold and playful"
            color = "vivid"
            transition = "quick cuts"
        else:
            style = "cinematic emotional"
            tone = "soft and expressive"
            color = "warm"
            transition = "gentle fades"
        duration = 18 if pacing == "fast" else 24 if pacing == "moderate" else 30
        return VideoIntent(
            pacing=pacing,
            visual_style=style,
            caption_tone=tone,
            transition_preference=transition,
            color_treatment=color,
            target_duration_seconds=duration,
            must_include=[],
            must_avoid=[],
        )

    def _caption_for(self, intent: VideoIntent, image: ImageAnalysis, index: int) -> str:
        if intent.pacing == "fast":
            templates = ["Set the pace", "Keep it moving", "The energy rises", "Big smiles", "One more beat"]
        elif intent.pacing == "slow":
            templates = ["A warm beginning", "Quiet joy", "Held in memory", "Soft light", "A lasting moment"]
        else:
            templates = ["The story begins", "Shared moments", "A gentle turn", "Closer now", "The final frame"]
        caption = templates[min(index - 1, len(templates) - 1)]
        if image.people >= 3 and index == 1 and intent.pacing != "fast":
            return "Together, in motion"
        return caption

    def _transition_for(self, intent: VideoIntent, index: int) -> str:
        if intent.pacing == "fast":
            return "quick-cut" if index % 2 == 0 else "slide"
        if intent.pacing == "slow":
            return "fade"
        return "soft-fade"

    def _animation_for(self, intent: VideoIntent, index: int) -> str:
        if intent.pacing == "fast":
            return "zoom-in"
        if index == 1:
            return "slow-zoom"
        return "subtle-pan"

    def _too_similar(self, candidate: ImageAnalysis, selected: list[ImageAnalysis]) -> bool:
        if not selected:
            return False
        for other in selected:
            hash_distance = self._hamming_distance(candidate.perceptual_hash, other.perceptual_hash)
            if (
                hash_distance < 10
                and candidate.people == other.people
                and candidate.indoor_outdoor == other.indoor_outdoor
                and candidate.palette_family == other.palette_family
            ):
                return True
        return False

    def _cluster_images(self, analyses: list[ImageAnalysis]) -> list[dict[str, Any]]:
        ordered = sorted(analyses, key=lambda a: (a.rank_score, a.confidence, a.quality_score), reverse=True)
        clusters: list[dict[str, Any]] = []
        for item in ordered:
            best_cluster = None
            best_score = 0.0
            for cluster in clusters:
                score = self._event_similarity(item, cluster["representative"])
                if score > best_score:
                    best_score = score
                    best_cluster = cluster
            if best_cluster is None or best_score < 0.62:
                clusters.append(
                    {
                        "representative": item,
                        "items": [item],
                        "event_types": {item.event_type},
                        "palette_families": {item.palette_family},
                    }
                )
            else:
                best_cluster["items"].append(item)
                best_cluster["event_types"].add(item.event_type)
                best_cluster["palette_families"].add(item.palette_family)
                if item.rank_score > best_cluster["representative"].rank_score:
                    best_cluster["representative"] = item
        for cluster in clusters:
            cluster["size"] = len(cluster["items"])
            cluster["total_rank"] = sum(item.rank_score for item in cluster["items"])
            cluster["avg_similarity"] = self._cluster_avg_similarity(cluster["items"])
            rep = cluster["representative"]
            cluster["label"] = self._cluster_label(rep, cluster["size"])
            cluster["event_types"] = sorted(cluster["event_types"])
            cluster["palette_families"] = sorted(cluster["palette_families"])
        return clusters

    def _event_similarity(self, a: ImageAnalysis, b: ImageAnalysis) -> float:
        hash_similarity = 1.0 - min(self._hamming_distance(a.perceptual_hash, b.perceptual_hash), 64) / 64.0
        palette_similarity = 1.0 if a.palette_family == b.palette_family else 0.0
        indoor_similarity = 1.0 if a.indoor_outdoor == b.indoor_outdoor else 0.0
        emotion_similarity = 1.0 if a.emotion == b.emotion else 0.0
        people_similarity = 1.0 - min(abs(a.people - b.people), 4) / 4.0
        event_similarity = 1.0 if a.event_type == b.event_type and a.event_type != "general" else 0.0
        return (
            0.42 * hash_similarity
            + 0.18 * palette_similarity
            + 0.12 * indoor_similarity
            + 0.10 * emotion_similarity
            + 0.08 * people_similarity
            + 0.10 * event_similarity
        )

    def _cluster_avg_similarity(self, items: list[ImageAnalysis]) -> float:
        if len(items) < 2:
            return 1.0
        scores = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                scores.append(self._event_similarity(items[i], items[j]))
        return sum(scores) / len(scores) if scores else 1.0

    def _cluster_label(self, item: ImageAnalysis, size: int) -> str:
        parts = [item.event_type, item.palette_family, item.indoor_outdoor]
        core = "-".join(part for part in parts if part and part != "general")
        return f"{core or 'general'}-{size}"

    def _hamming_distance(self, a: str, b: str) -> int:
        try:
            return bin(int(a, 16) ^ int(b, 16)).count("1")
        except Exception:
            return 64
