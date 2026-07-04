from __future__ import annotations

from pathlib import Path
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.graph.pipeline import run_pipeline
from app.services.input_adapter import InputAdapter


def make_demo_images(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    palette = [(220, 70, 70), (60, 140, 230), (80, 180, 120), (200, 180, 60)]
    labels = ["Arrival", "Smile", "Toast", "Dance"]
    for idx, (color, label) in enumerate(zip(palette, labels), start=1):
        img = Image.new("RGB", (960, 540), color)
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((36, 36, 924, 504), radius=30, outline=(255, 255, 255), width=6)
        draw.text((72, 72), f"Scene {idx}", fill=(255, 255, 255))
        draw.text((72, 132), label, fill=(255, 255, 255))
        img.save(folder / f"demo_{idx}.png")
    return folder


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    demo_images = make_demo_images(root / "sample_input" / "demo_images")
    adapter = InputAdapter()
    source_type, source_ref, images = adapter.load(str(demo_images))
    state = run_pipeline(
        source_type,
        source_ref,
        "Cinematic wedding reel, slow and emotional, warm tones, minimal text",
        images,
        output_root=root / "sample_output",
    )
    print(f"Sample output written to: {state.artifact_paths.run_dir}")


if __name__ == "__main__":
    main()
