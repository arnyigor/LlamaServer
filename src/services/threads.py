"""Фоновые потоки для сканирования моделей и обновления llama.cpp."""

import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from PySide6.QtCore import QThread, Signal

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

    def run(self):
        try:
            if not self.server_path.exists():
                raise FileNotFoundError(
                    f"llama-server.exe not found: {self.server_path}"
                )
            target_dir = self.server_path.parent
            current_build = self.get_current_build()
            release = self.fetch_latest_release()
            latest_build = self.parse_build_number(release.get("tag_name", ""))
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
            assets = self.select_assets(release)
            if not assets:
                raise RuntimeError(
                    "No Windows release asset found in the latest release"
                )
            with tempfile.TemporaryDirectory(prefix="llamacpp-update-") as temp_dir:
                temp_path = Path(temp_dir)
                extract_dir = temp_path / "extract"
                extract_dir.mkdir()
                for index, asset in enumerate(assets, start=1):
                    name = asset["name"]
                    archive_path = temp_path / name
                    self.progress.emit(f"Downloading {name} ({index}/{len(assets)})")
                    self.download(asset["browser_download_url"], archive_path)
                    self.progress.emit(f"Extracting {name}")
                    self.safe_extract_zip(archive_path, extract_dir)
                install_root = self.find_install_root(extract_dir)
                if not (install_root / "llama-server.exe").exists():
                    raise RuntimeError(
                        "Downloaded archive does not contain llama-server.exe"
                    )
                self.progress.emit(f"Installing into {target_dir}")
                self.copy_tree_contents(install_root, target_dir)
            self.percent.emit(100)
            self.completed.emit(True, f"Updated llama.cpp to build {latest_build}")
        except Exception as exc:
            self.completed.emit(False, f"Update failed: {exc}")

    def get_current_build(self):
        try:
            result = subprocess.run(
                [str(self.server_path), "--version"],
                capture_output=True,
                text=True,
                timeout=20,
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
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"GitHub API HTTP {e.code}: {e.reason}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error: {e.reason}") from e

    def parse_build_number(self, value):
        match = re.search(r"b?(\d+)", value or "")
        return int(match.group(1)) if match else None

    def select_assets(self, release):
        assets = release.get("assets", [])
        # Priority order: CUDA 12.4 > any CUDA > Vulkan > AVX2 > generic Windows
        patterns = [
            (re.compile(r"^llama-b\d+-bin-win-cuda-12\.4-x64\.zip$"), "CUDA 12.4"),
            (re.compile(r"^llama-b\d+-bin-win-cuda-.*\.zip$"), "CUDA"),
            (re.compile(r"^llama-b\d+-bin-win-vulkan-.*\.zip$"), "Vulkan"),
            (re.compile(r"^llama-b\d+-bin-win-avx2-.*\.zip$"), "AVX2"),
            (re.compile(r"^llama-b\d+-bin-win-.*\.zip$"), "Windows"),
        ]
        for pattern, label in patterns:
            for asset in assets:
                if pattern.match(asset.get("name", "")):
                    self.progress.emit(f"Selected {label} build: {asset['name']}")
                    return [asset]
        return []

    def download(self, url, destination):
        request = urllib.request.Request(url, headers={"User-Agent": "LlamaServerGUI"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                with open(destination, "wb") as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if total:
                            self.percent.emit(min(99, int(done * 100 / total)))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.reason}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error: {e.reason}") from e

    def safe_extract_zip(self, archive_path, destination):
        destination = destination.resolve()
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                try:
                    target.relative_to(destination)
                except ValueError as exc:
                    raise RuntimeError(f"Unsafe zip entry: {member.filename}")
            archive.extractall(destination)

    def find_install_root(self, extract_dir):
        candidates = sorted(extract_dir.rglob("llama-server.exe"))
        return candidates[0].parent if candidates else extract_dir

    def copy_tree_contents(self, source, destination):
        destination.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            target = destination / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
