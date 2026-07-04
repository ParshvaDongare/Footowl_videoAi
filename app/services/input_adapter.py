from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from app.models import ImageRecord, SourceType
from app.services.drive_service import GoogleDriveFolderDownloader


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class InputAdapter:
    def load(self, source_ref: str) -> tuple[SourceType, str, list[ImageRecord]]:
        path = Path(source_ref)
        if path.exists() and path.is_dir():
            print(f"[input] loading local folder {path}", flush=True)
            return SourceType.local_folder, str(path), self._load_local_images(path)
        if source_ref.startswith("https://") or source_ref.startswith("http://"):
            if "drive.google.com/drive/folders/" in source_ref:
                print("[input] loading public Google Drive folder", flush=True)
                downloader = GoogleDriveFolderDownloader()
                folder = downloader.download_public_folder(source_ref)
                return SourceType.google_drive_folder, source_ref, self._load_local_images(folder)
            raise ValueError("Only public Google Drive folder URLs are supported in the remote adapter.")
        raise FileNotFoundError(f"Source not found: {source_ref}")

    def _load_local_images(self, folder: Path) -> list[ImageRecord]:
        images: list[ImageRecord] = []
        for file in sorted(folder.iterdir()):
            if file.suffix.lower() not in SUPPORTED_EXTS:
                continue
            try:
                from PIL import Image

                with Image.open(file) as img:
                    width, height = img.size
            except Exception:
                width = height = 0
            images.append(
                ImageRecord(
                    image_id=file.stem,
                    path=str(file),
                    width=width,
                    height=height,
                    file_size=file.stat().st_size,
                )
            )
        return images
