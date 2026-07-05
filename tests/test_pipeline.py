from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw

from app.agents.nodes import PipelineServices
from app.graph.pipeline import run_pipeline
from app.models import CompileDiagnostic
from app.services.ai_service import AIService
from app.services.input_adapter import InputAdapter


def make_sample_images(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for idx, color in enumerate([(220, 70, 70), (60, 140, 230), (80, 180, 120), (200, 180, 60)], start=1):
        img = Image.new("RGB", (640, 360), color)
        draw = ImageDraw.Draw(img)
        # Add distinct visual content per image so perceptual hashes differ.
        draw.text((20, 20), f"IMG{idx}", fill=(255, 255, 255))
        draw.rectangle([100 * idx, 80, 100 * idx + 80, 160], fill=(255 - idx * 40, idx * 40, 128))
        draw.ellipse([200, 100 + idx * 20, 350, 200 + idx * 20], fill=(idx * 60, 200 - idx * 20, 100))
        img.save(folder / f"img{idx}.png")
    return folder


def make_two_event_images(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    warm_palette = [(220, 80, 80), (210, 60, 50), (200, 90, 60), (180, 70, 70)]
    cool_palette = [(60, 130, 220), (50, 110, 200)]
    for idx, color in enumerate(warm_palette, start=1):
        img = Image.new("RGB", (640, 360), color)
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), f"W{idx}", fill=(255, 255, 255))
        draw.rectangle([60 + idx * 20, 90, 140 + idx * 20, 180], fill=(255, 240, 240))
        img.save(folder / f"warm_{idx}.png")
    for idx, color in enumerate(cool_palette, start=1):
        img = Image.new("RGB", (640, 360), color)
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), f"C{idx}", fill=(255, 255, 255))
        draw.rectangle([260 - idx * 15, 120, 360 - idx * 15, 210], fill=(235, 245, 255))
        img.save(folder / f"cool_{idx}.png")
    return folder


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.image_dir = make_sample_images(self.tmp_path / "images")
        self.adapter = InputAdapter()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_pipeline_generates_artifacts(self):
        source_type, source_ref, images = self.adapter.load(str(self.image_dir))
        state = run_pipeline(
            source_type,
            source_ref,
            "Cinematic wedding reel, slow and emotional, warm tones, minimal text",
            images,
            output_root=self.tmp_path / "out1",
        )
        self.assertIsNotNone(state.storyboard)
        self.assertIsNotNone(state.composition_spec)
        self.assertTrue(state.remotion_code)
        self.assertIsNotNone(state.artifact_paths)
        self.assertIsNotNone(state.artifact_paths.storyboard_json)
        self.assertIsNotNone(state.artifact_paths.tsx_script)

        # Regression guard for P1: captions must not contain raw image_id strings
        # (i.e. no image_summary leaking into caption text).
        image_ids = {img.image_id for img in state.images}
        for scene in state.storyboard.scenes:
            for image_id in image_ids:
                self.assertNotIn(
                    image_id,
                    scene.caption,
                    msg=f"Raw image_id '{image_id}' leaked into caption: '{scene.caption}'",
                )

        # P3 regression guard: generated TSX must contain the assetPaths map.
        self.assertIn("assetPaths", state.remotion_code)

    def test_prompt_changes_storyboard(self):
        source_type, source_ref, images = self.adapter.load(str(self.image_dir))
        slow = run_pipeline(
            source_type,
            source_ref,
            "Cinematic wedding reel, slow and emotional, warm tones, minimal text",
            images,
            output_root=self.tmp_path / "slow",
        )
        fast = run_pipeline(
            source_type,
            source_ref,
            "Upbeat birthday reel, fast cuts, bold captions, energetic",
            images,
            output_root=self.tmp_path / "fast",
        )
        self.assertNotEqual(slow.storyboard.logline, fast.storyboard.logline)
        self.assertNotEqual(slow.video_intent.pacing, fast.video_intent.pacing)
        self.assertNotEqual(slow.storyboard.scenes[0].transition, fast.storyboard.scenes[0].transition)

    def test_judge_and_validation(self):
        prompt = "Clean corporate highlights, professional tone, subtle transitions"
        source_type, source_ref, images = self.adapter.load(str(self.image_dir))
        state = run_pipeline(
            source_type,
            source_ref,
            prompt,
            images,
            output_root=self.tmp_path / "out3",
        )
        self.assertIsNotNone(state.storyboard)
        self.assertGreaterEqual(len(state.storyboard.scenes), 1)

        # Now actually exercise judge_storyboard and assert a usable coherence score.
        ai = AIService()
        result = ai.judge_storyboard(state.storyboard, prompt)
        self.assertIn("coherence_score", result)
        self.assertGreaterEqual(result["coherence_score"], 0.6)

    def test_llm_as_judge_mocked(self):
        class FakeJudgeLLM:
            available = True

            def generate_json(self, *, model, system_prompt, user_prompt, response_model, temperature=0.0, image_paths=None):
                self.last_model = model
                self.last_prompt = user_prompt
                return response_model(coherence_score=0.95, notes=["mocked structured judge"])

        source_type, source_ref, images = self.adapter.load(str(self.image_dir))
        state = run_pipeline(
            source_type,
            source_ref,
            "Cinematic wedding reel, slow and emotional, warm tones, minimal text",
            images,
            output_root=self.tmp_path / "judge_mock",
        )
        ai = AIService(llm=FakeJudgeLLM())
        result = ai.judge_storyboard(state.storyboard, "Cinematic wedding reel, slow and emotional, warm tones, minimal text")
        self.assertGreaterEqual(result["coherence_score"], 0.9)
        self.assertIn("mocked structured judge", result["notes"][0])

    def test_compile_retry_path(self):
        class RetryOnceCompiler:
            def __init__(self):
                self.calls = 0

            def compile(self, tsx_code, spec, asset_paths):
                self.calls += 1
                if self.calls == 1:
                    return type(
                        "CompileResult",
                        (),
                        {
                            "ok": False,
                            "diagnostics": [CompileDiagnostic(category="syntax", message="synthetic syntax failure")],
                        },
                    )()
                return type("CompileResult", (), {"ok": True, "diagnostics": []})()\

            def classify(self, diagnostics):
                return "syntax_error"

        class RetryServices(PipelineServices):
            def __init__(self):
                super().__init__()
                self.compiler = RetryOnceCompiler()

        source_type, source_ref, images = self.adapter.load(str(self.image_dir))
        state = run_pipeline(
            source_type,
            source_ref,
            "Cinematic wedding reel, slow and emotional, warm tones, minimal text",
            images,
            output_root=self.tmp_path / "retry",
            services=RetryServices(),
        )
        self.assertEqual(state.retry_count, 1)
        self.assertTrue(state.output_video)

    def test_no_duplicate_images_selected(self):
        """Selected images must not have identical perceptual hashes (hamming > 0)."""
        source_type, source_ref, images = self.adapter.load(str(self.image_dir))
        state = run_pipeline(
            source_type,
            source_ref,
            "Cinematic wedding reel, slow and emotional, warm tones, minimal text",
            images,
            output_root=self.tmp_path / "dedup",
        )
        ai = AIService()
        selected = state.selected_images
        for i, a in enumerate(selected):
            for b in selected[i + 1:]:
                dist = ai._hamming_distance(a.perceptual_hash, b.perceptual_hash)
                self.assertGreater(
                    dist,
                    0,
                    msg=(
                        f"Images '{a.image_id}' and '{b.image_id}' have identical perceptual hashes "
                        f"and should not both be selected."
                    ),
                )

    def test_single_event_cluster_selected(self):
        """Mixed albums should collapse to one coherent event cluster in the reel."""
        mixed_dir = make_two_event_images(self.tmp_path / "mixed")
        source_type, source_ref, images = self.adapter.load(str(mixed_dir))
        state = run_pipeline(
            source_type,
            source_ref,
            "Cinematic wedding reel, slow and emotional, warm tones, minimal text",
            images,
            output_root=self.tmp_path / "mixed_out",
        )
        selected_palette_families = {img.palette_family for img in state.selected_images}
        self.assertEqual(
            len(selected_palette_families),
            1,
            msg=f"Selected images should belong to one event cluster, got: {selected_palette_families}",
        )
        self.assertTrue(state.selected_event_cluster)


if __name__ == "__main__":
    unittest.main()
