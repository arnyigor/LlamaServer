"""Фоновые потоки для сканирования моделей и обновления llama.cpp."""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import List

from PySide6.QtCore import QThread, Signal

from src.core.gguf_parser import extract_model_info, is_projector_file


class ModelScanner(QThread):
    """Поток для сканирования GGUF моделей."""

    models_found = Signal(list)
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, base_path):
        super().__init__()
        self.base_path = base_path
        self.setTerminationEnabled(False)

    def run(self):
        models: List[dict] = []
        base = Path(self.base_path)

        if not base.exists():
            self.error.emit(f"Папка не найдена: {base}")
            return

        try:
            all_files = [f for f in base.rglob("*.gguf") if not is_projector_file(f)]
            total = len(all_files)

            for i, gguf_file in enumerate(all_files, 1):
                if self.isInterruptionRequested():
                    self.progress.emit("⏹ Сканирование отменено")
                    models.sort(key=lambda x: x["display"].lower())
                    self.models_found.emit(models)
                    return

                try:
                    rel_path = gguf_file.relative_to(base)
                    info = extract_model_info(gguf_file)
                    info["display"] = str(rel_path)
                    models.append(info)
                except Exception as e:
                    self.progress.emit(f"⚠️ Пропуск {gguf_file.name}: {e}")
                    continue

                if i % 25 == 0 or i == total:
                    self.progress.emit(
                        f"Сканирование: {i}/{total}, найдено: {len(models)}"
                    )

        except PermissionError as e:
            self.error.emit(f"Нет доступа: {e}")

        models.sort(key=lambda x: x["display"].lower())
        self.models_found.emit(models)


class LlamaCppUpdater(QThread):
    """Поток для обновления llama.cpp."""

    progress = Signal(str)
    percent = Signal(int)
    completed = Signal(bool, str)
    API_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

    def __init__(self, server_path, cuda_version="12"):
        super().__init__()
        self.server_path = Path(server_path)
        self.cuda_version = cuda_version

    def run(self):
        try:
            release = self.fetch_latest_release()
            latest_build = self.parse_build_number(release.get("tag_name", ""))
            if latest_build is None:
                raise RuntimeError(
                    f"Cannot parse release tag: {release.get('tag_name')}"
                )

            assets = self.select_assets(release)
            if not assets:
                raise RuntimeError(
                    "No Windows release asset found in the latest release"
                )

            target_dir = self.resolve_target_dir(assets)
            server_exe = target_dir / "llama-server.exe"
            current_build = self.get_current_build(server_exe)
            current_text = (
                current_build if current_build is not None else "not installed"
            )
            self.progress.emit(
                f"llama.cpp local build: {current_text}, latest: {latest_build}"
            )
            if current_build is not None and current_build >= latest_build:
                self.percent.emit(100)
                self.completed.emit(False, f"Already up to date: build {current_build}")
                return

            if target_dir.exists() and any(target_dir.glob("*.exe")):
                self.progress.emit("Creating backup...")
                backup_path = self.backup_binaries(target_dir)
                self.progress.emit(f"Backup created: {backup_path.name}")
            else:
                target_dir.mkdir(parents=True, exist_ok=True)
                self.progress.emit(f"Installing new llama.cpp into {target_dir}")

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
                # Merge cudart DLLs from other extracted subdirs if any
                for subdir in extract_dir.iterdir():
                    if subdir.is_dir() and subdir.resolve() != install_root.resolve():
                        # Copy any .dll files from sibling dirs (cudart etc.)
                        for dll in subdir.rglob("*.dll"):
                            dest = target_dir / dll.name
                            if not dest.exists():
                                self.progress.emit(f"Installing DLL: {dll.name}")
                                shutil.copy2(dll, dest)
            self.percent.emit(100)
            self.completed.emit(True, f"Updated llama.cpp to build {latest_build}")
        except Exception as exc:
            self.completed.emit(False, f"Update failed: {exc}")

    def get_current_build(self, server_path=None):
        server_path = Path(server_path or self.server_path)
        if not server_path.exists():
            return None
        try:
            result = subprocess.run(
                [str(server_path), "--version"],
                capture_output=True,
                text=True,
                timeout=20,
                cwd=str(server_path.parent),
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
        # Сначала ищем формат bXXXX (стандартный для llama.cpp)
        match = re.search(r"b(\d+)", value or "")
        if match:
            num = int(match.group(1))
            return num if num >= 1 else None
        # Fallback: число в конце строки (для тегов без префикса)
        match = re.search(r"(\d+)$", value or "")
        if match:
            num = int(match.group(1))
            return num if num >= 1 else None
        return None

    def resolve_target_dir(self, assets):
        """Возвращает папку установки для выбранного архива.

        Если передан путь к exe или к папке с llama-server.exe — обновляем её.
        Если передана базовая папка, создаём/обновляем подпапку вида
        llama-win-cuda-12.4-x64 или llama-win-cuda-13.3-x64.
        """
        source = self.server_path
        if source.suffix.lower() == ".exe":
            return source.parent
        if source.is_dir() and (source / "llama-server.exe").exists():
            return source

        for asset in assets:
            name = asset.get("name", "")
            match = re.match(r"^llama-b\d+-bin-win-cuda-(\d+\.\d+)-x64\.zip$", name)
            if match:
                return source / f"llama-win-cuda-{match.group(1)}-x64"
            if re.match(r"^llama-b\d+-bin-win-vulkan-x64\.zip$", name):
                return source / "llama-win-vulkan-x64"
            if re.match(r"^llama-b\d+-bin-win-cpu-x64\.zip$", name):
                return source / "llama-win-cpu-x64"
        return source / f"llama-win-cuda-{self.cuda_version}-x64"

    def select_assets(self, release):
        assets = release.get("assets", [])
        cv = self.cuda_version  # "12" or "13"
        result = []

        # 1) Find main binaries for the selected CUDA major version
        #    Pattern: llama-b{build}-bin-win-cuda-{major}.{minor}-x64.zip
        bin_pattern = re.compile(
            rf"^llama-b\d+-bin-win-cuda-{re.escape(cv)}\.\d+-x64\.zip$"
        )
        bin_asset = None
        for asset in assets:
            if bin_pattern.match(asset.get("name", "")):
                bin_asset = asset
                break

        if bin_asset:
            result.append(bin_asset)
            self.progress.emit(f"Selected CUDA {cv} build: {bin_asset['name']}")
        else:
            # Fallback: try any CUDA asset matching the major version prefix
            fallback = re.compile(rf"^llama-b\d+-bin-win-cuda-{re.escape(cv)}.*\.zip$")
            for asset in assets:
                if fallback.match(asset.get("name", "")):
                    bin_asset = asset
                    result.append(asset)
                    self.progress.emit(
                        f"Selected CUDA {cv} build (fallback): {asset['name']}"
                    )
                    break

        if bin_asset:
            # 2) Find cudart DLLs for the selected CUDA major version
            #    Pattern: cudart-llama-bin-win-cuda-{major}.{minor}-x64.zip
            cudart_pattern = re.compile(
                rf"^cudart-llama-bin-win-cuda-{re.escape(cv)}\.\d+-x64\.zip$"
            )
            for asset in assets:
                if cudart_pattern.match(asset.get("name", "")):
                    result.append(asset)
                    self.progress.emit(
                        f"Selected CUDA {cv} runtime DLLs: {asset['name']}"
                    )
                    break
            return result

        # 3) If no CUDA build found at all, fallback to Vulkan → CPU
        if not result:
            fallback_patterns = [
                (re.compile(r"^llama-b\d+-bin-win-vulkan-x64\.zip$"), "Vulkan"),
                (re.compile(r"^llama-b\d+-bin-win-cpu-x64\.zip$"), "CPU x64"),
            ]
            for pattern, label in fallback_patterns:
                for asset in assets:
                    if pattern.match(asset.get("name", "")):
                        result.append(asset)
                        self.progress.emit(
                            f"CUDA {cv} not available, fallback to {label}: {asset['name']}"
                        )
                        return result

        return result

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

    @staticmethod
    def backup_binaries(target_dir: Path, keep: int = 5) -> Path:
        """Создание бэкапа бинарников с ограничением количества."""
        backup_dir = target_dir / "backup"
        backup_dir.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_subdir = backup_dir / f"backup_{timestamp}"
        backup_subdir.mkdir(exist_ok=True)

        for pattern in ["*.exe", "*.dll"]:
            for file in target_dir.glob(pattern):
                try:
                    shutil.copy2(file, backup_subdir / file.name)
                except OSError:
                    pass

        # Оставляем только последние `keep` бэкапов
        backups = sorted(
            [
                d
                for d in backup_dir.iterdir()
                if d.is_dir() and d.name.startswith("backup_")
            ],
            key=lambda d: d.name,
            reverse=True,
        )
        for old in backups[keep:]:
            shutil.rmtree(old, ignore_errors=True)

        return backup_subdir
