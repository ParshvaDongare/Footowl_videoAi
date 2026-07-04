from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys

import cv2
import numpy as np
from PIL import Image

from app.models import ImageAnalysis, ImageRecord
from app.utils import clamp


@dataclass
class VisionService:
    def analyze(self, record: ImageRecord) -> ImageAnalysis:
        path = Path(record.path)
        image = cv2.imread(str(path))
        if image is None:
            return ImageAnalysis(
                image_id=record.image_id,
                image_summary="Unreadable image",
                people=0,
                objects=[],
                palette_family="unknown",
                emotion="neutral",
                blur_score=0.0,
                quality_score=0.0,
                indoor_outdoor="unknown",
                aesthetic_score=0.0,
                duplicate_score=0.0,
                confidence=0.1,
                rank_score=0.0,
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_norm = clamp(blur_score / 400.0, 0.0, 1.0)
        brightness = float(gray.mean() / 255.0)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = float(hsv[:, :, 1].mean() / 255.0)
        quality_score = clamp(0.45 * blur_norm + 0.35 * brightness + 0.2 * saturation, 0.0, 1.0)
        aesthetic_score = clamp(0.5 * saturation + 0.5 * (1.0 - abs(0.55 - brightness)), 0.0, 1.0)
        indoor_outdoor = "outdoor" if brightness > 0.48 else "indoor"
        emotion = self._emotion_from_light(brightness, saturation)
        objects = self._object_hints(record.image_id, path.name)
        event_type = self._event_type_from_objects(objects)
        palette_family = self._palette_family(image)
        people = self._person_estimate(str(path))
        image_summary = self._summary(record.image_id, people, emotion, indoor_outdoor, objects, palette_family)
        # duplicate_score is 0.0 per-image; real dedup happens via perceptual hash in AIService._too_similar
        duplicate_score = 0.0
        confidence = clamp(0.6 + 0.2 * quality_score + 0.1 * aesthetic_score, 0.0, 1.0)
        rank_score = clamp(0.35 * quality_score + 0.25 * aesthetic_score + 0.2 * blur_norm + 0.2 * confidence, 0.0, 1.0)
        perceptual_hash = self._average_hash(gray)
        return ImageAnalysis(
            image_id=record.image_id,
            image_summary=image_summary,
            perceptual_hash=perceptual_hash,
            palette_family=palette_family,
            people=people,
            objects=objects,
            event_type=event_type,
            emotion=emotion,
            blur_score=blur_score,
            quality_score=quality_score,
            indoor_outdoor=indoor_outdoor,
            aesthetic_score=aesthetic_score,
            duplicate_score=duplicate_score,
            confidence=confidence,
            rank_score=rank_score,
        )

    def _person_estimate(self, image_path: str) -> int:
        try:
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
            return int(len(faces))
        except Exception as exc:
            print(f"[vision] face detection failed for {image_path}: {exc}", file=sys.stderr)
            return 0

    def _object_hints(self, image_id: str, filename: str) -> list[str]:
        tokens = {*(re.findall(r"[a-zA-Z]+", image_id.lower())), *(re.findall(r"[a-zA-Z]+", filename.lower()))}
        hints = []
        if {"wedding", "bride", "groom", "bridal", "mehendi", "haldi", "sangeet"} & tokens:
            hints.append("wedding")
        if {"birthday", "cake", "party", "celebration"} & tokens:
            hints.append("party")
        if {"corporate", "office", "team", "conference", "meeting"} & tokens:
            hints.append("office")
        if {"stage", "dance", "music", "concert", "performance"} & tokens:
            hints.append("performance")
        if {"golf", "soccer", "football", "cricket", "tennis", "sport", "match", "game", "player", "athlete"} & tokens:
            hints.append("sport")
        return hints

    def _event_type_from_objects(self, objects: list[str]) -> str:
        """Derive a single canonical event type from detected object hints."""
        priority = ["wedding", "party", "performance", "sport", "office"]
        for p in priority:
            if p in objects:
                return p
        return "general"

    def _emotion_from_light(self, brightness: float, saturation: float) -> str:
        if brightness > 0.65 and saturation > 0.4:
            return "joyful"
        if brightness < 0.35:
            return "moody"
        if saturation < 0.2:
            return "calm"
        return "warm"

    def _summary(self, image_id: str, people: int, emotion: str, indoor_outdoor: str, objects: list[str], palette_family: str) -> str:
        object_text = ", ".join(objects) if objects else "no strong object cue"
        subject = "group" if people >= 3 else "couple" if people == 2 else "solo moment" if people == 1 else "scene"
        return f"{image_id}: {subject}, {emotion}, {indoor_outdoor}, {palette_family}, {object_text}"

    def _palette_family(self, image_bgr) -> str:
        mean_bgr = image_bgr.reshape(-1, 3).mean(axis=0)
        b, g, r = [float(v) / 255.0 for v in mean_bgr]
        dominance = max(r, g, b)
        if dominance < 0.25:
            return "dark"
        if r > g + 0.08 and r > b + 0.08:
            return "warm"
        if b > r + 0.08 and b > g + 0.08:
            return "cool"
        if g > r + 0.08 and g > b + 0.08:
            return "green"
        return "neutral"


    def _average_hash(self, gray: np.ndarray) -> str:
        resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        mean = resized.mean()
        bits = resized > mean
        value = 0
        for bit in bits.flatten():
            value = (value << 1) | int(bool(bit))
        return f"{value:016x}"
