from __future__ import annotations

from dataclasses import dataclass
from math import sin, pi
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from app.models import CompositionSpec


def _ease_in_out(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


@dataclass
class RenderingService:
    def render(self, spec: CompositionSpec, output_path: Path, asset_lookup: dict[str, str], intent: Any | None = None) -> Path:
        width, height, fps = spec.width, spec.height, spec.fps
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )
        title_font, caption_font, meta_font = self._load_fonts()
        transition_frames = max(8, int(fps * 0.35))

        scene_sources = [self._load_image(asset_lookup.get(scene.image_id), width, height) for scene in spec.scenes]
        scene_sources = [img if img is not None else Image.new("RGB", (width, height), (12, 14, 18)) for img in scene_sources]

        for index, scene in enumerate(spec.scenes):
            image_path = asset_lookup.get(scene.image_id)
            frames = max(1, int(round(scene.duration_seconds * fps)))
            base = self._load_image(image_path, width, height) or Image.new("RGB", (width, height), (12, 14, 18))
            next_img = scene_sources[index + 1] if index + 1 < len(scene_sources) else None
            for frame_index in range(frames):
                progress = frame_index / max(frames - 1, 1)
                frame = self._render_frame(
                    scene=scene,
                    base=base,
                    next_img=next_img,
                    progress=progress,
                    width=width,
                    height=height,
                    title_font=title_font,
                    caption_font=caption_font,
                    meta_font=meta_font,
                    transition_frames=transition_frames,
                    intent=intent,
                )
                writer.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        writer.release()
        return output_path

    def _load_image(self, image_path: str | None, width: int, height: int) -> Image.Image | None:
        if not image_path or not Path(image_path).exists():
            return None
        img = Image.open(image_path).convert("RGB")
        return img.resize((width, height), Image.Resampling.LANCZOS)

    def _load_fonts(self):
        candidates = [
            "C:/Windows/Fonts/seguiemj.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                try:
                    return (
                        ImageFont.truetype(candidate, 30),
                        ImageFont.truetype(candidate, 64),
                        ImageFont.truetype(candidate, 22),
                    )
                except Exception:
                    pass
        default = ImageFont.load_default()
        return default, default, default

    def _render_frame(
        self,
        scene,
        base: Image.Image,
        next_img: Image.Image | None,
        progress: float,
        width: int,
        height: int,
        title_font,
        caption_font,
        meta_font,
        transition_frames: int,
        intent: Any | None,
    ) -> Image.Image:
        progress = max(0.0, min(1.0, progress))
        canvas = self._background(base, progress, width, height, intent)
        draw = ImageDraw.Draw(canvas, "RGBA")

        card_margin_x = int(width * 0.055)
        card_margin_top = int(height * 0.06)
        card_margin_bottom = int(height * 0.16)
        card_box = (
            card_margin_x,
            card_margin_top,
            width - card_margin_x,
            height - card_margin_bottom,
        )

        card_alpha = 255
        sharp = self._foreground_frame(base, scene.animation, progress, card_box[2] - card_box[0], card_box[3] - card_box[1], intent)
        shadow_box = (card_box[0] + 10, card_box[1] + 12, card_box[2] + 10, card_box[3] + 12)
        self._rounded_rect(draw, shadow_box, 34, fill=(0, 0, 0, 85))
        canvas = self._paste_rounded(canvas, sharp, card_box, 34, alpha=card_alpha)

        if next_img is not None and progress > 1.0 - (transition_frames / max(transition_frames * 2, 1)):
            blend_t = (progress - (1.0 - (transition_frames / max(transition_frames * 2, 1)))) / max(transition_frames / max(transition_frames * 2, 1), 0.001)
            blend_t = max(0.0, min(1.0, blend_t))
            next_bg = self._background(next_img, 0.0, width, height, intent)
            canvas = Image.blend(canvas, next_bg, blend_t * 0.35)

        draw = ImageDraw.Draw(canvas, "RGBA")

        # Dark readability overlays.
        self._vertical_gradient(draw, width, height)
        self._top_header(draw, width, intent, title_font, meta_font)
        self._bottom_caption(draw, width, height, scene.caption, scene.transition, progress, caption_font, meta_font, scene.timing)
        self._progress_bar(draw, width, height, progress, intent)
        return canvas

    def _background(self, image: Image.Image, progress: float, width: int, height: int, intent: Any | None) -> Image.Image:
        cover = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        blur_radius = 18 if getattr(intent, "pacing", "moderate") != "fast" else 14
        blurred = cover.filter(ImageFilter.GaussianBlur(blur_radius))
        overlay = Image.new("RGBA", (width, height), self._accent_overlay(intent))
        blended = Image.alpha_composite(blurred.convert("RGBA"), overlay)
        return blended.convert("RGB")

    def _foreground_frame(self, image: Image.Image, animation: str, progress: float, width: int, height: int, intent: Any | None) -> Image.Image:
        # Ken Burns motion against a rounded "photo card" inside the blurred background.
        scale_map = {
            "slow-zoom": (1.10, 1.00),
            "zoom-in": (1.14, 1.02),
            "subtle-pan": (1.06, 1.00),
        }
        start_scale, end_scale = scale_map.get(animation, (1.08, 1.00))
        scale = start_scale + (end_scale - start_scale) * _ease_in_out(progress)
        scaled_w = max(width, int(image.width * scale))
        scaled_h = max(height, int(image.height * scale))
        resized = image.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

        max_x = max(0, scaled_w - width)
        max_y = max(0, scaled_h - height)
        pan_x = int(max_x * (0.45 + 0.08 * sin(progress * pi * 2.0)))
        pan_y = int(max_y * (0.45 + 0.04 * sin(progress * pi)))
        if getattr(intent, "pacing", "moderate") == "fast":
            pan_x = int(max_x * (0.2 + 0.6 * progress))
            pan_y = int(max_y * 0.1)
        crop = resized.crop((pan_x, pan_y, pan_x + width, pan_y + height))
        return crop

    def _top_header(self, draw: ImageDraw.ImageDraw, width: int, intent: Any | None, title_font, meta_font) -> None:
        title = getattr(intent, "visual_style", "ReelGraph")
        caption = getattr(intent, "caption_tone", "curated moments")
        draw.rounded_rectangle((42, 34, 360, 108), radius=24, fill=(0, 0, 0, 115), outline=(255, 255, 255, 35), width=1)
        draw.text((62, 46), "FotoOwl", fill=(240, 240, 240, 230), font=meta_font)
        draw.text((62, 70), title.title(), fill=(255, 255, 255, 255), font=title_font)
        draw.text((62, 118), caption.title(), fill=(225, 225, 225, 220), font=meta_font)

    def _bottom_caption(self, draw: ImageDraw.ImageDraw, width: int, height: int, caption: str, transition: str, progress: float, caption_font, meta_font, timing: str) -> None:
        band_h = int(height * 0.26)
        band_y0 = height - band_h
        self._vertical_gradient(draw, width, height, start_alpha=35, end_alpha=220, y0=band_y0)
        draw.rounded_rectangle((42, band_y0 + 16, width - 42, height - 28), radius=30, fill=(8, 10, 14, 150), outline=(255, 255, 255, 28), width=1)
        # Anchor caption text to a fixed position inside the bottom band — never overlaps progress bar
        caption_y = band_y0 + 32
        lines = self._wrap_text(caption, caption_font, width - 160, max_lines=2)
        self._draw_multiline(draw, (72, caption_y), lines, caption_font, fill=(255, 255, 255, 255), line_gap=8)

    def _progress_bar(self, draw: ImageDraw.ImageDraw, width: int, height: int, progress: float, intent: Any | None) -> None:
        bar_y = height - 24
        draw.rounded_rectangle((72, bar_y, width - 72, bar_y + 8), radius=4, fill=(255, 255, 255, 38))
        accent = self._accent_color(intent)
        fill_w = int((width - 144) * progress)
        draw.rounded_rectangle((72, bar_y, 72 + fill_w, bar_y + 8), radius=4, fill=accent + (220,))

    def _vertical_gradient(self, draw: ImageDraw.ImageDraw, width: int, height: int, start_alpha: int = 0, end_alpha: int = 180, y0: int = 0) -> None:
        if end_alpha <= start_alpha:
            return
        grad_h = height - y0
        for i in range(grad_h):
            alpha = int(start_alpha + (end_alpha - start_alpha) * (i / max(grad_h - 1, 1)))
            draw.line([(0, y0 + i), (width, y0 + i)], fill=(0, 0, 0, alpha))

    def _rounded_rect(self, draw: ImageDraw.ImageDraw, box, radius: int, fill, outline=None, width: int = 1) -> None:
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

    def _paste_rounded(self, canvas: Image.Image, image: Image.Image, box, radius: int, alpha: int = 255) -> Image.Image:
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        cropped = ImageOps.fit(image, (w, h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        if alpha < 255:
            cropped = cropped.convert("RGBA")
            cropped.putalpha(alpha)
        else:
            cropped = cropped.convert("RGBA")
        mask = Image.new("L", (w, h), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rounded_rectangle((0, 0, w, h), radius=radius, fill=255)
        canvas_rgba = canvas.convert("RGBA")
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        layer.paste(cropped, (x0, y0), mask)
        canvas_rgba.alpha_composite(layer)
        return canvas_rgba.convert("RGB")

    def _draw_multiline(self, draw: ImageDraw.ImageDraw, origin, lines: list[str], font, fill, line_gap: int = 6) -> None:
        x, y = origin
        cursor = y
        for line in lines:
            draw.text((x, cursor), line, fill=fill, font=font)
            bbox = draw.textbbox((x, cursor), line, font=font)
            cursor += (bbox[3] - bbox[1]) + line_gap

    def _wrap_text(self, text: str, font, max_width: int, max_lines: int = 2) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            trial = word if not current else f"{current} {word}"
            if self._text_width(trial, font) <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = lines[-1].rstrip(" .,") + "..."
        return lines

    def _text_width(self, text: str, font) -> int:
        dummy = Image.new("RGB", (10, 10))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]

    def _accent_color(self, intent: Any | None) -> tuple[int, int, int]:
        style = getattr(intent, "color_treatment", "").lower()
        pacing = getattr(intent, "pacing", "moderate")
        if "warm" in style:
            return (244, 173, 66)
        if "vivid" in style:
            return (82, 190, 255)
        if pacing == "fast":
            return (255, 90, 140)
        return (183, 148, 255)

    def _accent_overlay(self, intent: Any | None) -> tuple[int, int, int, int]:
        r, g, b = self._accent_color(intent)
        return (r, g, b, 22)
