from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.models import CompileDiagnostic, CompositionScene, CompositionSpec, ImageAnalysis, ScenePlan, Storyboard, VideoIntent, ValidationResult
from app.utils import clamp
from app.services.prompt_loader import load_prompt


@dataclass
class AIService:
    def __post_init__(self) -> None:
        # Load version-controlled prompt templates.  These are the system prompts
        # that will be passed to real LLMs when model backends are wired in.
        self.prompt_intent: str = load_prompt("intent")
        self.prompt_storyboard: str = load_prompt("storyboard")
        self.prompt_composition: str = load_prompt("composition")
        self.prompt_fix: str = load_prompt("fix")

    def parse_intent(self, prompt: str) -> VideoIntent:
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

    def select_images(
        self,
        analyses: list[ImageAnalysis],
        intent: VideoIntent,
        max_selected_images: int,
    ) -> tuple[list[ImageAnalysis], dict[str, Any]]:
        clusters = self._cluster_images(analyses)
        if not clusters:
            return [], {"selected_event_cluster": "", "event_clusters": []}

        # Pick the most coherent cluster, not just the highest-ranked individual shots.
        best_cluster = max(clusters, key=lambda c: (c["total_rank"], c["size"], c["avg_similarity"]))
        ordered = sorted(best_cluster["items"], key=lambda a: (a.rank_score, a.confidence, a.quality_score), reverse=True)

        selected: list[ImageAnalysis] = []
        used_signatures: set[tuple[str, str, str]] = set()
        for item in ordered:
            signature = (item.people.__str__(), item.indoor_outdoor, item.emotion)
            if self._too_similar(item, selected):
                continue
            if signature in used_signatures and len(selected) < max_selected_images - 1:
                continue
            used_signatures.add(signature)
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
        intro = f"{intent.visual_style.title()} reel with {intent.caption_tone} captions."
        scenes: list[ScenePlan] = []
        total = max(intent.target_duration_seconds, len(selected_images) * 3)
        base = total / max(len(selected_images), 1)
        for index, image in enumerate(selected_images, start=1):
            duration = clamp(base * (1.2 if index == 1 else 1.0), 2.0, 6.0)
            scenes.append(
                ScenePlan(
                    scene_id=f"scene_{index}",
                    image_id=image.image_id,
                    duration_seconds=duration,
                    caption=self._caption_for(intent, image, index),
                    transition=self._transition_for(intent, index),
                    animation=self._animation_for(intent, index),
                    timing="opening" if index == 1 else "mid" if index < len(selected_images) else "ending",
                    rationale=f"Selected for {image.image_summary} and {style_note}",
                )
            )
        storyboard = Storyboard(
            title=f"{intent.visual_style.title()} Highlight Reel",
            logline=intro,
            tone=intent.caption_tone,
            scenes=scenes[:],
            notes=[style_note, f"Selected {len(scenes)} scenes from {len(selected_images)} candidate images."],
        )
        return storyboard

    def plan_composition(
        self,
        storyboard: Storyboard,
        width: int,
        height: int,
        fps: int,
    ) -> CompositionSpec:
        scenes: list[CompositionScene] = []
        start = 0.0
        for scene in storyboard.scenes:
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

    def generate_tsx(self, spec: CompositionSpec, asset_lookup: dict[str, str] | None = None) -> str:
        lines = [
            "import React from 'react';",
            "import {AbsoluteFill, Img, interpolate, useCurrentFrame, useVideoConfig} from 'remotion';",
            "",
        ]
        # Embed real asset paths so the TSX resolves against the local file system.
        if asset_lookup:
            lines.append("const assetPaths: Record<string, string> = {")
            for image_id, path in asset_lookup.items():
                # Use forward-slash paths for cross-platform compatibility inside TSX strings.
                safe_path = path.replace("\\", "/")
                lines.append(f"  {image_id!r}: {safe_path!r},")
            lines.append("};")  
        else:
            lines.append("const assetPaths: Record<string, string> = {};")
        lines.append("")
        lines.append("const scenes = [")
        for scene in spec.scenes:
            lines.append(
                f"  {{ id: '{scene.scene_id}', imageId: '{scene.image_id}', start: {scene.start_seconds:.2f}, "
                f"duration: {scene.duration_seconds:.2f}, caption: {scene.caption!r}, transition: {scene.transition!r}, "
                f"animation: {scene.animation!r} }},"
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
                "        <div style={{fontSize: 20, opacity: 0.85}}>fps {fps} · {width}x{height}</div>",
                "      </AbsoluteFill>",
                "    </AbsoluteFill>",
                "  );",
                "};",
            ]
        )
        return "\n".join(lines) + "\n"

    def repair_storyboard_for_error(
        self, storyboard: Storyboard, diagnostics: list[CompileDiagnostic]
    ) -> Storyboard:
        """Category-aware storyboard repair based on compiler diagnostics.

        - asset  : remove the specific scene(s) referencing the missing image_id.
        - syntax / structure / export : no storyboard change — the fix comes from
          re-running generate_tsx with the same spec.
        """
        categories = {d.category for d in diagnostics}
        if "asset" in categories:
            # Collect image_ids mentioned in asset diagnostics.
            missing_ids: set[str] = set()
            for d in diagnostics:
                if d.category == "asset":
                    # message format: "Missing scene asset reference: <image_id>"
                    parts = d.message.split(":")
                    if len(parts) >= 2:
                        missing_ids.add(parts[-1].strip())
            if missing_ids and len(storyboard.scenes) > len(missing_ids):
                storyboard.scenes = [s for s in storyboard.scenes if s.image_id not in missing_ids]
        # For syntax / structure / export errors the storyboard is not at fault;
        # generate_tsx will be re-called with the (unchanged) spec.
        return storyboard

    def judge_storyboard(self, storyboard: Storyboard, prompt: str) -> dict[str, Any]:
        keywords = set(prompt.lower().split())
        score = 0.6
        if any(word in storyboard.logline.lower() for word in keywords):
            score += 0.2
        if len(storyboard.scenes) >= 2:
            score += 0.1
        if storyboard.scenes and storyboard.scenes[0].timing == "opening":
            score += 0.1
        return {"coherence_score": min(score, 1.0), "notes": ["Storyboard matches prompt intent and has an opening/mid/end arc."]}

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
