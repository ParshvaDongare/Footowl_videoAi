from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request

from app.utils import ensure_dir


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class GoogleDriveFolderDownloader:
    cache_root: Path | None = None

    def download_public_folder(self, folder_url: str) -> Path:
        folder_id = self._extract_folder_id(folder_url)
        print(f"[drive] resolving public folder {folder_id}", flush=True)
        cache_root = ensure_dir(self.cache_root or Path(tempfile.gettempdir()) / "fotovowl_drive_cache")
        folder_dir = ensure_dir(cache_root / folder_id)
        if any(folder_dir.iterdir()):
            print(f"[drive] using cached folder {folder_dir}", flush=True)
            return folder_dir
        print("[drive] fetching folder page", flush=True)
        html = self._fetch(folder_url)
        file_ids = self._extract_file_ids(html)
        if not file_ids:
            raise ValueError(
                "Unable to discover downloadable files in the shared Google Drive folder. "
                "Make sure the folder is public and contains image files."
            )
        print(f"[drive] discovered {len(file_ids)} file references", flush=True)
        for file_id in file_ids:
            try:
                print(f"[drive] downloading {file_id}", flush=True)
                self._download_file(file_id, folder_dir)
            except Exception:
                continue
        if not any(folder_dir.iterdir()):
            raise ValueError(
                "No downloadable images were found in the shared Google Drive folder. "
                "If the folder is public, try a smaller folder or copy the images locally."
            )
        print(f"[drive] cached downloads in {folder_dir}", flush=True)
        return folder_dir

    def _extract_folder_id(self, url: str) -> str:
        match = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
        if not match:
            raise ValueError(f"Not a valid Google Drive folder URL: {url}")
        return match.group(1)

    def _fetch(self, url: str) -> str:
        request = urllib.request.Request(url + ("&" if "?" in url else "?") + "hl=en", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=12) as response:
            return response.read().decode("utf-8", errors="ignore")

    def _extract_file_ids(self, html: str) -> list[str]:
        patterns = [
            r'/file/d/([a-zA-Z0-9_-]+)',
            r'"fileId":"([a-zA-Z0-9_-]+)"',
            r'data-id="([a-zA-Z0-9_-]+)"',
        ]
        file_ids: list[str] = []
        for pattern in patterns:
            for match in re.findall(pattern, html):
                if match not in file_ids:
                    file_ids.append(match)
        return file_ids

    def _download_file(self, file_id: str, folder_dir: Path) -> None:
        base = f"https://drive.google.com/uc?export=download&id={file_id}"
        content, filename = self._download_with_confirm(base)
        if filename:
            dest = folder_dir / filename
        else:
            dest = folder_dir / f"{file_id}.bin"
        if dest.suffix.lower() not in IMAGE_EXTS and not self._looks_like_image(filename or dest.name):
            return
        dest.write_bytes(content)

    def _download_with_confirm(self, url: str) -> tuple[bytes, str | None]:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=20) as response:
            content = response.read()
            content_type = response.headers.get("Content-Type", "")
            disposition = response.headers.get("Content-Disposition", "")
            filename = self._filename_from_disposition(disposition)
            if content_type.startswith("text/html"):
                html = content.decode("utf-8", errors="ignore")
                confirm = self._extract_confirm_token(html)
                if confirm:
                    return self._download_with_confirm(url + f"&confirm={confirm}")
                raise ValueError("Drive download returned HTML instead of a file.")
            return content, filename

    def _extract_confirm_token(self, html: str) -> str | None:
        match = re.search(r'confirm=([0-9A-Za-z_]+)', html)
        return match.group(1) if match else None

    def _filename_from_disposition(self, disposition: str) -> str | None:
        match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', disposition)
        if match:
            name = urllib.parse.unquote(match.group(1))
            return os.path.basename(name)
        return None

    def _looks_like_image(self, name: str) -> bool:
        return Path(name).suffix.lower() in IMAGE_EXTS
