from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import os
import tempfile

from app.models import ImageRecord, SourceType
from app.services.drive_service import GoogleDriveFolderDownloader
from app.utils import ensure_dir


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@dataclass
class InputAdapter:
    def load(self, source_ref: str) -> tuple[SourceType, str, list[ImageRecord]]:
        path = Path(source_ref)
        if path.exists() and path.is_dir():
            print(f"[input] loading local folder {path}", flush=True)
            return SourceType.local_folder, str(path), self._load_local_media(path)
        if source_ref.startswith("https://") or source_ref.startswith("http://"):
            if "drive.google.com/drive/folders/" in source_ref:
                print("[input] loading public Google Drive folder", flush=True)
                downloader = GoogleDriveFolderDownloader()
                folder = downloader.download_public_folder(source_ref)
                return SourceType.google_drive_folder, source_ref, self._load_local_media(folder)
            raise ValueError("Only public Google Drive folder URLs are supported in the remote adapter.")
        raise FileNotFoundError(f"Source not found: {source_ref}")

    def _load_local_media(self, folder: Path) -> list[ImageRecord]:
        images: list[ImageRecord] = []
        frame_root = ensure_dir(Path(tempfile.gettempdir()) / "fotovowl_frames" / hashlib.sha1(str(folder).encode("utf-8")).hexdigest()[:12])
        for file in sorted(folder.iterdir()):
            suffix = file.suffix.lower()
            if suffix in SUPPORTED_IMAGE_EXTS:
                images.extend(self._load_local_images(file))
            elif suffix in SUPPORTED_VIDEO_EXTS:
                images.extend(self._extract_video_frames(file, frame_root))
                continue
        return images

    def _load_local_images(self, file: Path) -> list[ImageRecord]:
        try:
            from PIL import Image

            with Image.open(file) as img:
                width, height = img.size
        except Exception:
            width = height = 0
        return [
            ImageRecord(
                image_id=file.stem,
                path=str(file),
                width=width,
                height=height,
                file_size=file.stat().st_size,
            )
        ]

    def _extract_video_frames(self, file: Path, frame_root: Path) -> list[ImageRecord]:
        try:
            import cv2

            capture = cv2.VideoCapture(str(file))
            if not capture.isOpened():
                return []
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count <= 0:
                frame_indexes = [0]
            else:
                frame_indexes = sorted({max(0, int(frame_count * 0.25)), max(0, int(frame_count * 0.75))})
            results: list[ImageRecord] = []
            for idx, frame_index in enumerate(frame_indexes, start=1):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                out = frame_root / f"{file.stem}_frame{idx:02d}.png"
                cv2.imwrite(str(out), frame)
                height, width = frame.shape[:2]
                results.append(
                    ImageRecord(
                        image_id=f"{file.stem}_frame{idx:02d}",
                        path=str(out),
                        width=width,
                        height=height,
                        file_size=out.stat().st_size,
                    )
                )
            capture.release()
            return results
        except Exception:
            return []
