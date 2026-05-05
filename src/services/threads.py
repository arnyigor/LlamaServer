"""Фоновые потоки для сканирования моделей и обновления llama.cpp."""

import json
import os
import re
import shutil
import ssl
import subprocess
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Union

from PySide6.QtCore import QThread, Signal

from src.core.constants import (
    DOWNLOAD_CHUNK_SIZE,
    TIMEOUT_DOWNLOAD,
    TIMEOUT_GITHUB_API,
    TIMEOUT_VERSION_CHECK,
)
from src.core.gguf_parser import extract_model_info, is_projector_file


class ModelScanner(QThread):
    """Поток для сканирования GGUF моделей."""

    models_found = Signal(list)
    progress = Signal(str)

    def __init__(self, base_path):
        super().__init__()
        self.base_path = base_path

    def run(self):
        models = []
        base = Path(self.base_path)
        if base.exists():
            for gguf_file in base.rglob("*.gguf"):
                if self.isInterruptionRequested():
                    break
                if is_projector_file(gguf_file):
                    continue
                rel_path = gguf_file.relative_to(base)
                info = extract_model_info(gguf_file)
                info["display"] = str(rel_path)
                models.append(info)
                if len(models) % 25 == 0:
                    self.progress.emit(f"Найдено моделей: {len(models)}")
        models.sort(key=lambda i: i["display"].lower())
        self.models_found.emit(models)


class LlamaCppUpdater(QThread):
    """Поток для обновления llama.cpp."""

    progress = Signal(str)
    percent = Signal(int)
    completed = Signal(bool, str)
    API_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

    def __init__(self, server_path):
        super().__init__()
        self.server_path = Path(server_path)
        self._is_running = False

    def run(self):
        self._is_running = True
        try:
            self.progress.emit("DEBUG: Updater thread started")

            if not self.server_path.exists():
                raise FileNotFoundError(
                    f"llama-server.exe not found: {self.server_path}"
                )
            target_dir = self.server_path.parent

            # Проверка прав на запись
            if not os.access(target_dir, os.W_OK):
                raise PermissionError(f"No write access to: {target_dir}")

            self.progress.emit("Checking local version...")
            current_build = self.get_current_build()
            self.progress.emit(f"DEBUG: Current build = {current_build}")

            self.progress.emit("Connecting to GitHub API...")
            self.progress.emit(f"DEBUG: API URL = {self.API_URL}")
            release = self.fetch_latest_release()
            self.progress.emit(f"DEBUG: Release fetched successfully")

            latest_build = self.parse_build_number(release.get("tag_name", ""))
            self.progress.emit(
                f"DEBUG: Latest build = {latest_build}, tag = {release.get('tag_name', 'N/A')}"
            )

            if latest_build is None:
                raise RuntimeError(
                    f"Cannot parse release tag: {release.get('tag_name')}"
                )
            current_text = current_build if current_build is not None else "unknown"
            self.progress.emit(
                f"llama.cpp local build: {current_text}, latest: {latest_build}"
            )
            if current_build is not None and current_build >= latest_build:
                self.percent.emit(100)
                self.completed.emit(False, f"Already up to date: build {current_build}")
                return

            self.progress.emit("Selecting download assets...")
            assets = self.select_assets(release)
            self.progress.emit(f"DEBUG: Found {len(assets)} assets")

            if not assets:
                available = [
                    a.get("name", "unknown") for a in release.get("assets", [])[:10]
                ]
                raise RuntimeError(
                    f"No Windows x64 release asset found. Available assets: {available}"
                )

            self.progress.emit(f"Found {len(assets)} asset(s) to download")

            with tempfile.TemporaryDirectory(prefix="llamacpp-update-") as temp_dir:
                temp_path = Path(temp_dir)
                extract_dir = temp_path / "extract"
                extract_dir.mkdir()
                for index, asset in enumerate(assets, start=1):
                    if self.isInterruptionRequested():
                        raise InterruptedError("Update cancelled by user")
                    name = asset["name"]
                    archive_path = temp_path / name
                    self.progress.emit(f"Downloading {name} ({index}/{len(assets)})")
                    self.progress.emit(
                        f"DEBUG: URL = {asset.get('browser_download_url', 'N/A')}"
                    )
                    self.download(asset["browser_download_url"], archive_path)
                    self.progress.emit(f"Extracting {name}")
                    self.safe_extract_zip(archive_path, extract_dir)

                self.progress.emit("Looking for llama-server.exe in archive...")
                install_root = self.find_install_root(extract_dir)
                if not (install_root / "llama-server.exe").exists():
                    raise RuntimeError(
                        "Downloaded archive does not contain llama-server.exe"
                    )

                self.progress.emit(f"Installing into {target_dir}")
                self.copy_tree_contents(install_root, target_dir)

            self.percent.emit(100)
            self.completed.emit(True, f"Updated llama.cpp to build {latest_build}")
        except InterruptedError:
            self.progress.emit("Update cancelled")
            self.completed.emit(False, "Update cancelled by user")
        except Exception as exc:
            import traceback

            error_msg = f"Update failed: {exc}\n{traceback.format_exc()}"
            self.progress.emit(f"ERROR: {error_msg}")
            self.completed.emit(False, f"Update failed: {exc}")
        finally:
            self._is_running = False

    def isRunning(self):
        return self._is_running

    def get_current_build(self):
        try:
            result = subprocess.run(
                [str(self.server_path), "--version"],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_VERSION_CHECK,
                cwd=str(self.server_path.parent),
                check=False,
            )
        except Exception as exc:
            self.progress.emit(f"Cannot read local version: {exc}")
            return None
        text = f"{result.stdout}\n{result.stderr}"
        match = re.search(r"version:\s*(\d+)", text, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def fetch_latest_release(self):
        request = urllib.request.Request(
            self.API_URL,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "LlamaServerGUI",
            },
        )
        self.progress.emit("Checking latest llama.cpp release")
        try:
            with urllib.request.urlopen(
                request, timeout=TIMEOUT_GITHUB_API
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
                self.progress.emit(f"Found release: {data.get('tag_name', 'unknown')}")
                return data
        except urllib.error.HTTPError as e:
            self.progress.emit(f"GitHub API error: {e.code} {e.reason}")
            raise
        except urllib.error.URLError as e:
            self.progress.emit(f"Network error: {e.reason}")
            raise

    def parse_build_number(self, value):
        """Парсинг номера билда из тега релиза.

        Ожидает формат 'b1234' или '1234'.
        Не парсит произвольные числа (например, 'v1.0').
        """
        if not value:
            return None
        # Ищем b1234 или просто 1234 в начале/конце строки
        match = re.search(r"[Bb]?(\d{3,})", value)
        if match:
            num = int(match.group(1))
            # Проверяем, что это похоже на билд (обычно > 1000)
            if num >= 1000:
                return num
        return None

    def select_assets(self, release):
        """Выбор подходящих ассетов для Windows.

        Ищет CUDA 12.4 билд, если не найден — любой Windows x64 билд.
        """
        assets = release.get("assets", [])

        # Сначала ищем CUDA 12.4
        pattern_cuda = re.compile(
            r"^llama-b\d+-bin-win-cuda-12\.4-x64\.zip$", re.IGNORECASE
        )
        for asset in assets:
            if pattern_cuda.match(asset.get("name", "")):
                return [asset]

        # Затем любой Windows x64 CUDA
        pattern_win = re.compile(r"^llama-b\d+-bin-win-.*-x64\.zip$", re.IGNORECASE)
        for asset in assets:
            if pattern_win.match(asset.get("name", "")):
                return [asset]

        # Логируем доступные ассеты для диагностики
        available = [a.get("name", "unknown") for a in assets[:10]]
        self.progress.emit(f"Available assets: {available}")

        return []

    def download(self, url: str, destination: Union[str, Path]) -> None:
        """Скачивание файла с валидацией URL и ограничением размера."""
        parsed = urllib.parse.urlparse(url)
        if parsed.hostname not in (
            "github.com",
            "objects.githubusercontent.com",
            "githubusercontent.com",
        ):
            raise ValueError(f"URL must be from GitHub: {url}")

        ssl_context = ssl.create_default_context()
        request = urllib.request.Request(url, headers={"User-Agent": "LlamaServerGUI"})
        with urllib.request.urlopen(
            request, timeout=TIMEOUT_DOWNLOAD, context=ssl_context
        ) as response:
            total = int(response.headers.get("Content-Length") or 0)
            MAX_DOWNLOAD_SIZE = 2 * 1024 * 1024 * 1024
            if total > MAX_DOWNLOAD_SIZE:
                raise RuntimeError(
                    f"File too large: {total} bytes (max: {MAX_DOWNLOAD_SIZE})"
                )

            done = 0
            with open(destination, "wb") as out:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if done > MAX_DOWNLOAD_SIZE:
                        raise RuntimeError(
                            f"Download exceeded maximum size: {MAX_DOWNLOAD_SIZE}"
                        )
                    if total:
                        self.percent.emit(min(99, int(done * 100 / total)))

    def safe_extract_zip(
        self, archive_path: Union[str, Path], destination: Union[str, Path]
    ) -> None:
        """Безопасная распаковка ZIP с защитой от Zip Slip и symlink-атак.

        Вместо extractall() извлекает файлы вручную, предотвращая:
        - Path traversal через ..
        - Symlink-атаки
        - Абсолютные пути
        """
        destination = Path(destination).resolve()
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                # Проверяем на абсолютные пути и path traversal
                if member.filename.startswith("/") or ".." in member.filename:
                    raise RuntimeError(
                        f"Unsafe zip entry (absolute/path traversal): {member.filename}"
                    )

                target = (destination / member.filename).resolve()
                try:
                    target.relative_to(destination)
                except ValueError:
                    raise RuntimeError(
                        f"Unsafe zip entry (escapes destination): {member.filename}"
                    )

                # Проверяем на symlinks (до извлечения!)
                # ZipInfo.is_symlink() доступен только в Python 3.9+
                # Для совместимости проверяем external_attr
                is_symlink = (member.external_attr >> 28) == 0xA
                if is_symlink:
                    raise RuntimeError(
                        f"Symlinks not allowed in zip: {member.filename}"
                    )

                # Создаем родительские директории
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue

                # Извлекаем файл вручную (не через extractall)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    def find_install_root(self, extract_dir):
        candidates = sorted(extract_dir.rglob("llama-server.exe"))
        return candidates[0].parent if candidates else extract_dir

    def copy_tree_contents(self, source, destination):
        """Копирование содержимого директории без symlink-атак."""
        destination.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            target = destination / item.name
            # Пропускаем symlinks
            if item.is_symlink():
                continue
            if item.is_dir():
                self.copy_tree_contents(item, target)
            else:
                shutil.copy2(item, target)
