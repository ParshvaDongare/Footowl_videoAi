from __future__ import annotations

import argparse
from pathlib import Path
import json

from app.graph.pipeline import run_pipeline
from app.services.input_adapter import InputAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="FotoOwl ReelGraph pipeline")
    parser.add_argument("--source", required=True, help="Local folder containing images")
    parser.add_argument("--prompt", required=True, help="Creative brief")
    args = parser.parse_args()
    adapter = InputAdapter()
    print("[main] starting input adapter", flush=True)
    source_type, source_ref, images = adapter.load(args.source)
    print(f"[main] loaded {len(images)} images", flush=True)
    print("[main] running pipeline", flush=True)
    state = run_pipeline(source_type, source_ref, args.prompt, images)
    print(f"[main] completed with status={state.status.value}", flush=True)
    print(json.dumps(state.model_dump(), indent=2))


if __name__ == "__main__":
    main()
