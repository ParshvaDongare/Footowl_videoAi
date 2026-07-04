from __future__ import annotations


STYLE_GUIDES = [
    {
        "id": "cinematic",
        "text": (
            "Cinematic style: slower pacing, warm contrast, soft fades, emotional captions, "
            "gentle motion, and longer opening/closing beats."
        ),
    },
    {
        "id": "upbeat",
        "text": (
            "Upbeat style: fast pacing, punchy captions, energetic cuts, quick transitions, "
            "higher text density, and vivid color treatment."
        ),
    },
    {
        "id": "corporate",
        "text": (
            "Corporate style: clean pacing, restrained transitions, concise captions, subtle motion, "
            "balanced composition, and polished neutral color treatment."
        ),
    },
]


REMOTION_DOCS = {
    "remotion_components": [
        {
            "id": "absolute-fill",
            "text": "AbsoluteFill is the base container for each frame and supports inline style objects.",
        },
        {
            "id": "img-component",
            "text": "Img renders images by providing a src prop with a local or remote asset path.",
        },
    ],
    "remotion_animation": [
        {
            "id": "interpolate",
            "text": "interpolate maps a frame value to opacity, scale, and translation ranges.",
        },
        {
            "id": "spring",
            "text": "spring produces eased motion and can be used for entrance animations and micro transitions.",
        },
    ],
    "remotion_transition": [
        {
            "id": "fade",
            "text": "Fade transitions reduce opacity from one scene to the next and are safe for all reel styles.",
        },
        {
            "id": "slide",
            "text": "Slide transitions translate the next scene from an offset and work well for upbeat reels.",
        },
    ],
    "remotion_cli": [
        {
            "id": "render-media",
            "text": "Remotion render-media compiles a composition and renders it to MP4 when the environment is configured.",
        },
        {
            "id": "preview",
            "text": "Remotion preview starts a local preview server for interactive inspection of the composition.",
        },
    ],
}

