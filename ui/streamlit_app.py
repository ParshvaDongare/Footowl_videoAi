from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from app.graph.pipeline import run_pipeline
from app.services.input_adapter import InputAdapter


RATIOS = {
    "9:16": (720, 1280),
    "16:9": (1280, 720),
    "1:1": (1080, 1080),
}


def build_enhanced_prompt(prompt: str, style: str, transitions: str, animations: str, duration: int, captions: bool, music: str, voice_over: str, ratio: str) -> str:
    parts = [
        prompt.strip(),
        "",
        "Production preferences:",
        f"- Video style: {style}",
        f"- Transition preference: {transitions}",
        f"- Animation preference: {animations}",
        f"- Target duration: {duration} seconds",
        f"- Aspect ratio: {ratio}",
        f"- Captions/text overlays: {'enabled' if captions else 'disabled'}",
        f"- Background music: {music}",
        f"- Voice-over: {voice_over}",
        "",
        "Use these preferences as part of the creative brief and keep the reel coherent with the uploaded media.",
    ]
    return "\n".join(parts)


def main() -> None:
    st.set_page_config(page_title="FotoOwl ReelGraph", page_icon="🎬", layout="wide")
    st.title("FotoOwl ReelGraph")
    st.caption("Paste a public Google Drive folder link, describe the reel, and generate a video.")

    with st.sidebar:
        st.header("Optional Controls")
        ratio = st.selectbox("Aspect ratio", list(RATIOS.keys()), index=0)
        duration = st.slider("Target duration (seconds)", min_value=8, max_value=60, value=20, step=1)
        style = st.selectbox("Video style", ["cinematic", "vlog", "promotional", "aesthetic", "wedding", "birthday", "corporate"], index=0)
        transitions = st.selectbox("Transitions", ["auto", "fade", "soft fade", "quick cuts", "slide"], index=0)
        animations = st.selectbox("Animations", ["auto", "subtle pan", "slow zoom", "zoom in", "dynamic"], index=0)
        captions = st.checkbox("Text overlays / captions", value=True)
        music = st.selectbox("Background music", ["auto", "no music", "uplifting", "ambient", "cinematic", "energetic"], index=0)
        voice_over = st.selectbox("Voice-over", ["auto", "no voice-over", "narration", "dramatic", "friendly"], index=1)

    drive_link = st.text_input("Google Drive folder link", placeholder="https://drive.google.com/drive/folders/...")
    prompt = st.text_area(
        "Prompt",
        height=120,
        placeholder="Cinematic wedding reel, slow and emotional, warm tones, minimal text",
    )

    col_a, col_b = st.columns([1, 3])
    with col_a:
        generate = st.button("Generate video", type="primary")
    with col_b:
        st.write("If you only provide the Drive link and prompt, the AI will auto-select the sequence, motion, and pacing.")

    if generate:
        if not drive_link.strip():
            st.error("Please paste a Google Drive folder link.")
            st.stop()
        if not prompt.strip():
            st.error("Please enter a prompt.")
            st.stop()

        enhanced_prompt = build_enhanced_prompt(
            prompt=prompt,
            style=style,
            transitions=transitions,
            animations=animations,
            duration=duration,
            captions=captions,
            music=music,
            voice_over=voice_over,
            ratio=ratio,
        )
        width, height = RATIOS[ratio]
        adapter = InputAdapter()

        with st.spinner("Loading media from Drive and running the pipeline..."):
            source_type, source_ref, images = adapter.load(drive_link.strip())
            state = run_pipeline(
                source_type,
                source_ref,
                enhanced_prompt,
                images,
                video_width=width,
                video_height=height,
                target_duration_seconds=duration,
                output_root=Path("output") / "ui_runs",
            )

        st.success(f"Done. Status: {state.status.value}")
        st.subheader("Output")
        if state.output_video and Path(state.output_video).exists():
            st.video(state.output_video)
            with open(state.output_video, "rb") as f:
                st.download_button(
                    "Download MP4",
                    data=f.read(),
                    file_name="fotovowl_reel.mp4",
                    mime="video/mp4",
                )
        else:
            st.warning("No output video was produced.")

        left, right = st.columns(2)
        with left:
            st.subheader("Storyboard")
            st.json(state.storyboard.model_dump() if state.storyboard else {})
        with right:
            st.subheader("Composition")
            st.json(state.composition_spec.model_dump() if state.composition_spec else {})

        st.subheader("Pipeline Trace")
        st.json(state.model_dump())
        if state.artifact_paths:
            st.write("Artifacts")
            st.json(state.artifact_paths.model_dump())


if __name__ == "__main__":
    main()
