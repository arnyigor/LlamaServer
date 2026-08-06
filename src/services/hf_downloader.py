"""Асинхронный поиск и скачивание GGUF-файлов с Hugging Face."""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List

from PySide6.QtCore import QThread, Signal

from src.core.gguf_parser import is_mtp_draft_file, is_projector_file


HF_API_MODEL_URL = "https://huggingface.co/api/models/{repo_id}?blobs=true"
HF_RESOLVE_URL = "https://huggingface.co/{repo_id}/resolve/main/{filename}?download=true"
USER_AGENT = "LlamaServerGUI"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024

_QUANT_ORDER = [
    "IQ1",
    "Q2",
    "IQ2",
    "Q3",
    "IQ3",
    "Q4",
    "IQ4",
    "Q5",
    "IQ5",
    "Q6",
    "Q8",
    "F16",
    "BF16",
    "F32",
]


class HfRepoError(RuntimeError):
    """Ошибка при обращении к Hugging Face."""


class HfDownloadInterrupted(RuntimeError):
    """Остановка загрузки пользователем: pause/cancel."""


class HfRepoScanner(QThread):
    """Фоновый поток, который получает список GGUF-файлов репозитория."""

    completed = Signal(dict)
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, repo_or_url: str, quant_filter: str = ""):
        super().__init__()
        self.repo_or_url = repo_or_url
        self.quant_filter = quant_filter

    def run(self):
        try:
            repo_id = normalize_hf_repo_id(self.repo_or_url)
            self.progress.emit(f"Hugging Face: чтение списка файлов {repo_id}")
            files = fetch_gguf_files(repo_id)
            main_files = [f for f in files if not f.get("is_projector")]
            projector_files = [f for f in files if f.get("is_projector")]
            filtered = filter_model_files(main_files, self.quant_filter)
            self.completed.emit(
                {
                    "repo_id": repo_id,
                    "files": filtered,
                    "all_files": main_files,
                    "projectors": projector_files,
                    "filter": self.quant_filter,
                }
            )
        except Exception as exc:
            self.error.emit(str(exc))


class HfModelDownloader(QThread):
    """Фоновая загрузка выбранных GGUF в LM Studio-compatible структуру."""

    progress = Signal(str)
    percent = Signal(int)
    completed = Signal(bool, str)

    def __init__(self, repo_id: str, files: List[Dict], base_model_dir: str):
        super().__init__()
        self.repo_id = repo_id
        self.files = files
        self.base_model_dir = Path(base_model_dir)
        self.delete_partial_on_stop = False

    def pause(self):
        """Останавливает загрузку, сохраняя .part для последующей докачки."""
        self.delete_partial_on_stop = False
        self.requestInterruption()

    def cancel_and_delete(self):
        """Останавливает загрузку и удаляет частичные .part файлы выбранной загрузки."""
        self.delete_partial_on_stop = True
        self.requestInterruption()

    def run(self):
        try:
            if not self.files:
                raise HfRepoError("Не выбраны файлы для скачивания")
            target_root = lmstudio_repo_dir(self.base_model_dir, self.repo_id)
            target_root.mkdir(parents=True, exist_ok=True)

            total_files = len(self.files)
            total_bytes = sum(int(f.get("size") or 0) for f in self.files)
            completed_bytes = 0
            started_at = time.monotonic()
            for index, file_info in enumerate(self.files, 1):
                if self.isInterruptionRequested():
                    self._cleanup_all_partial_files(target_root)
                    self.completed.emit(False, self._stop_message())
                    return
                filename = str(file_info.get("rfilename") or file_info.get("name") or "")
                if not filename.lower().endswith(".gguf"):
                    continue
                target = safe_repo_file_path(target_root, filename)
                size = int(file_info.get("size") or 0)
                self.progress.emit(f"Скачивание {filename} ({index}/{total_files})")
                completed_bytes = self._download_one(
                    filename,
                    target,
                    index,
                    total_files,
                    size,
                    completed_bytes,
                    total_bytes,
                    started_at,
                )

            self.percent.emit(100)
            self.completed.emit(True, f"Готово: {target_root}")
        except HfDownloadInterrupted as exc:
            self.completed.emit(False, str(exc))
        except Exception as exc:
            self.completed.emit(False, f"Ошибка скачивания: {exc}")

    def _download_one(
        self,
        filename: str,
        target: Path,
        index: int,
        total_files: int,
        expected_size: int,
        completed_bytes: int,
        total_bytes: int,
        started_at: float,
    ) -> int:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and expected_size > 0 and target.stat().st_size == expected_size:
            completed_bytes += expected_size
            self.progress.emit(
                f"Уже скачано: {target.name} | всего {format_bytes(completed_bytes)}"
            )
            if total_bytes:
                self.percent.emit(min(99, int(completed_bytes * 100 / total_bytes)))
            else:
                self.percent.emit(min(99, int(index * 100 / total_files)))
            return completed_bytes

        part = target.with_suffix(target.suffix + ".part")
        if target.exists() and expected_size > 0:
            target_size = target.stat().st_size
            if 0 < target_size < expected_size and not part.exists():
                target.replace(part)
                self.progress.emit(
                    f"Найден недокачанный файл {target.name}: "
                    f"{format_bytes(target_size)} / {format_bytes(expected_size)}"
                )

        resume_from = part.stat().st_size if part.exists() else 0
        if expected_size > 0 and resume_from >= expected_size:
            part.replace(target)
            completed_bytes += expected_size
            self.percent.emit(min(99, int(completed_bytes * 100 / total_bytes)) if total_bytes else 99)
            return completed_bytes

        url = resolve_file_url(self.repo_id, filename)
        headers = {"User-Agent": USER_AGENT}
        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"
            self.progress.emit(
                f"Продолжение {filename}: уже есть "
                f"{format_bytes(resume_from)} / {format_bytes(expected_size)}"
            )
        request = urllib.request.Request(url, headers=headers)
        try:
            try:
                response_cm = urllib.request.urlopen(request, timeout=60)
            except urllib.error.HTTPError as exc:
                if exc.code == 416 and expected_size > 0 and part.exists() and part.stat().st_size >= expected_size:
                    part.replace(target)
                    completed_bytes += expected_size
                    self.percent.emit(min(99, int(completed_bytes * 100 / total_bytes)) if total_bytes else 99)
                    return completed_bytes
                raise

            with response_cm as response:
                status = int(getattr(response, "status", response.getcode()) or 0)
                can_resume = resume_from > 0 and status == 206
                if resume_from > 0 and not can_resume:
                    self.progress.emit(
                        f"Сервер не поддержал докачку для {filename}, начинаю файл заново"
                    )
                    resume_from = 0

                content_len = int(response.headers.get("Content-Length") or 0)
                file_total = expected_size or (resume_from + content_len if can_resume else content_len)
                if not total_bytes and file_total:
                    total_bytes = file_total * total_files
                done = resume_from if can_resume else 0
                last_emit = 0.0
                mode = "ab" if can_resume else "wb"
                with open(part, mode) as out:
                    while True:
                        if self.isInterruptionRequested():
                            out.flush()
                            if self.delete_partial_on_stop:
                                delete_file_safely(part)
                            raise HfDownloadInterrupted(self._stop_message(part))
                        chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        now = time.monotonic()
                        if now - last_emit >= 0.4 or done == file_total:
                            current_total_done = completed_bytes + done
                            percent = self._overall_percent(
                                index,
                                total_files,
                                done,
                                file_total,
                                current_total_done,
                                total_bytes,
                            )
                            self.percent.emit(percent)
                            self.progress.emit(
                                self._progress_text(
                                    filename,
                                    index,
                                    total_files,
                                    done,
                                    file_total,
                                    current_total_done,
                                    total_bytes,
                                    started_at,
                                )
                            )
                            last_emit = now
            part.replace(target)
            actual_size = target.stat().st_size if target.exists() else done
            return completed_bytes + (expected_size or actual_size)
        except urllib.error.HTTPError as exc:
            raise HfRepoError(f"HTTP {exc.code}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise HfRepoError(f"Сетевая ошибка: {exc.reason}") from exc
        except BaseException:
            if part.exists() and self.isInterruptionRequested() and self.delete_partial_on_stop:
                delete_file_safely(part)
            raise

    def _cleanup_all_partial_files(self, target_root: Path):
        if not self.delete_partial_on_stop:
            return
        for file_info in self.files:
            filename = str(file_info.get("rfilename") or file_info.get("name") or "")
            if not filename.lower().endswith(".gguf"):
                continue
            try:
                target = safe_repo_file_path(target_root, filename)
            except HfRepoError:
                continue
            delete_file_safely(target.with_suffix(target.suffix + ".part"))

    def _stop_message(self, part: Path | None = None) -> str:
        if self.delete_partial_on_stop:
            return "Скачивание отменено. Частичный .part файл удалён."
        if part:
            return f"Пауза: частичный файл сохранён для докачки: {part}"
        return "Пауза: частичный файл сохранён для докачки."

    def _overall_percent(
        self,
        index: int,
        total_files: int,
        done: int,
        file_total: int,
        current_total_done: int,
        total_bytes: int,
    ) -> int:
        if total_bytes:
            return min(99, int(current_total_done * 100 / total_bytes))
        if file_total:
            file_fraction = min(1.0, done / file_total)
            return min(99, int(((index - 1) + file_fraction) * 100 / total_files))
        return min(99, int((index - 1) * 100 / total_files))

    def _progress_text(
        self,
        filename: str,
        index: int,
        total_files: int,
        done: int,
        file_total: int,
        current_total_done: int,
        total_bytes: int,
        started_at: float,
    ) -> str:
        elapsed = max(0.001, time.monotonic() - started_at)
        speed = current_total_done / elapsed
        remaining = max(0, total_bytes - current_total_done) if total_bytes else 0
        eta = format_eta(remaining / speed) if remaining and speed > 0 else "?"
        file_total_text = format_bytes(file_total) if file_total else "размер неизвестен"
        if total_bytes:
            total_text = (
                f"всего {format_bytes(current_total_done)} / {format_bytes(total_bytes)}, "
                f"осталось {format_bytes(remaining)}, ETA {eta}"
            )
        else:
            total_text = "общий размер неизвестен"
        return (
            f"{filename} ({index}/{total_files}): "
            f"{format_bytes(done)} / {file_total_text}; "
            f"{total_text}; скорость {format_bytes(speed)}/s"
        )


def normalize_hf_repo_id(value: str) -> str:
    """Нормализует repo id из `owner/model` или URL Hugging Face."""
    text = str(value or "").strip().strip('"').strip("'")
    if not text:
        raise HfRepoError("Укажите Hugging Face repo id или URL")

    if text.startswith("hf://"):
        text = text[5:]

    if re.match(r"^https?://", text, re.IGNORECASE):
        parsed = urllib.parse.urlparse(text)
        if parsed.netloc.lower() not in {"huggingface.co", "www.huggingface.co"}:
            raise HfRepoError("Поддерживаются только ссылки huggingface.co")
        parts = [urllib.parse.unquote(p) for p in parsed.path.split("/") if p]
        if len(parts) >= 2 and parts[0] in {"models", "datasets", "spaces"}:
            parts = parts[1:]
        stop_words = {"tree", "blob", "resolve", "raw"}
        repo_parts = []
        for part in parts:
            if part in stop_words:
                break
            repo_parts.append(part)
            if len(repo_parts) == 2:
                break
        text = "/".join(repo_parts)

    text = text.strip("/")
    if len(text.split("/")) < 2:
        raise HfRepoError("Repo id должен быть вида author/model")
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$", text):
        raise HfRepoError(f"Некорректный Hugging Face repo id: {text}")
    return text


def fetch_gguf_files(repo_id: str) -> List[Dict]:
    url = HF_API_MODEL_URL.format(repo_id=urllib.parse.quote(repo_id, safe="/"))
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise HfRepoError(f"Репозиторий не найден: {repo_id}") from exc
        raise HfRepoError(f"Hugging Face API HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise HfRepoError(f"Сетевая ошибка: {exc.reason}") from exc

    siblings = payload.get("siblings") or []
    result = []
    for item in siblings:
        filename = str(item.get("rfilename") or "")
        if not filename.lower().endswith(".gguf"):
            continue
        size = item.get("size")
        if not isinstance(size, int):
            lfs = item.get("lfs") if isinstance(item.get("lfs"), dict) else {}
            size = lfs.get("size") if isinstance(lfs.get("size"), int) else 0
        if not size:
            size = fetch_remote_size(repo_id, filename)
        result.append(
            {
                "rfilename": filename,
                "name": PurePosixPath(filename).name,
                "size": size or 0,
                "size_text": format_bytes(size or 0),
                "quant": quant_hint(filename),
                "is_projector": is_projector_file(filename),
            }
        )
    result.sort(key=lambda f: (f.get("is_projector", False), str(f.get("name", "")).lower()))
    return result


def filter_model_files(files: Iterable[Dict], filter_text: str) -> List[Dict]:
    text = str(filter_text or "").strip().upper()
    if not text:
        return list(files)

    quant_range = quant_range_from_filter(text)
    if quant_range:
        return [f for f in files if quant_base(f.get("quant") or f.get("name") or "") in quant_range]

    tokens = [t for t in re.split(r"[\s,;]+", text) if t]
    if not tokens:
        return list(files)
    return [
        f
        for f in files
        if any(token in str(f.get("name", "")).upper() or token in str(f.get("quant", "")).upper() for token in tokens)
    ]


def quant_range_from_filter(text: str) -> set:
    names = "|".join(re.escape(q) for q in sorted(_QUANT_ORDER, key=len, reverse=True))
    match = re.search(rf"({names})\s*-\s*({names})", text)
    if not match:
        return set()
    start, end = match.group(1), match.group(2)
    a, b = _QUANT_ORDER.index(start), _QUANT_ORDER.index(end)
    if a > b:
        a, b = b, a
    return set(_QUANT_ORDER[a : b + 1])


def quant_base(value: str) -> str:
    text = str(value or "").upper()
    match = re.search(r"(BF16|F32|F16|IQ[1-5]|Q[2-8])", text)
    return match.group(1) if match else ""


def quant_hint(filename: str) -> str:
    name = PurePosixPath(filename).name.upper()
    match = re.search(
        r"(UD-)?(IQ[1-5]_[A-Z0-9_]+|Q[2-8]_[A-Z0-9_]+|Q[2-8]|BF16|F16|F32)",
        name,
    )
    return match.group(0) if match else ""


def lmstudio_repo_dir(base_model_dir: Path, repo_id: str) -> Path:
    author, model = repo_id.split("/", 1)
    return base_model_dir / author / model


def list_local_repo_files(base_model_dir: Path, repo_id: str) -> Dict:
    """Список локальных файлов HF repo в LM Studio-compatible папке."""
    root = lmstudio_repo_dir(Path(base_model_dir), repo_id)
    result = {"root": str(root), "exists": root.exists(), "files": [], "total_size": 0}
    if not root.exists():
        return result
    try:
        files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: str(p).lower())
    except OSError:
        return result
    for path in files:
        try:
            rel = path.relative_to(root).as_posix()
            size = path.stat().st_size
        except OSError:
            continue
        result["files"].append(
            {
                "path": str(path),
                "relative": rel,
                "name": path.name,
                "size": size,
                "size_text": format_bytes(size),
                "is_partial": path.name.lower().endswith(".part"),
            }
        )
        result["total_size"] += size
    result["total_size_text"] = format_bytes(result["total_size"])
    return result


def _folder_size(path: Path) -> int:
    total = 0
    try:
        files = path.rglob("*") if path.is_dir() else [path]
        for item in files:
            if item.is_file():
                try:
                    total += item.stat().st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


def list_all_local_model_entries(base_model_dir: Path) -> Dict:
    """List all local model folders/files under Models, not only HF downloads.

    Entries are intentionally based on detected main GGUF files.  Projectors and
    MTP/draft GGUF files are not shown as standalone models, but are included in
    folder size and will be deleted when their model folder is deleted.
    """
    root = Path(base_model_dir)
    result = {"root": str(root), "exists": root.exists(), "entries": [], "total_size": 0}
    if not root.exists() or not root.is_dir():
        return result

    try:
        ggufs = sorted(root.rglob("*.gguf"), key=lambda p: str(p).lower())
    except OSError:
        return result

    grouped: Dict[Path, Dict] = {}
    root_resolved = root.resolve()
    for gguf in ggufs:
        try:
            if not gguf.is_file() or is_projector_file(gguf) or is_mtp_draft_file(gguf):
                continue
            gguf.relative_to(root_resolved)
        except (OSError, ValueError):
            continue

        if gguf.parent.resolve() == root_resolved:
            key = gguf.resolve()
            entry_type = "file"
            target = gguf
        else:
            key = gguf.parent.resolve()
            entry_type = "folder"
            target = gguf.parent

        entry = grouped.setdefault(
            key,
            {
                "type": entry_type,
                "path": str(target),
                "relative": target.resolve().relative_to(root_resolved).as_posix(),
                "name": target.name,
                "gguf_count": 0,
                "examples": [],
                "size": 0,
                "size_text": "0 B",
            },
        )
        entry["gguf_count"] += 1
        if len(entry["examples"]) < 3:
            entry["examples"].append(gguf.name)

    entries = []
    for entry in grouped.values():
        target = Path(entry["path"])
        size = _folder_size(target)
        entry["size"] = size
        entry["size_text"] = format_bytes(size)
        result["total_size"] += size
        entries.append(entry)
    entries.sort(key=lambda item: str(item.get("relative", "")).lower())
    result["entries"] = entries
    result["total_size_text"] = format_bytes(result["total_size"])
    return result


def find_partial_downloads(base_model_dir: Path, repo_id: str) -> List[Dict]:
    """Возвращает локальные .gguf.part для репозитория без обращения к сети."""
    root = lmstudio_repo_dir(Path(base_model_dir), repo_id)
    if not root.exists():
        return []
    result = []
    try:
        parts = sorted(root.rglob("*.gguf.part"), key=lambda p: str(p).lower())
    except OSError:
        return []
    for part in parts:
        try:
            rel = part.relative_to(root).as_posix()
            filename = rel[:-5] if rel.endswith(".part") else rel
            size = part.stat().st_size
        except OSError:
            continue
        result.append(
            {
                "rfilename": filename,
                "name": PurePosixPath(filename).name,
                "partial_path": str(part),
                "partial_size": size,
                "partial_size_text": format_bytes(size),
            }
        )
    return result


def partial_download_info(
    base_model_dir: Path, repo_id: str, filename: str, expected_size: int = 0
) -> Dict:
    """Информация о .part для конкретного файла репозитория."""
    try:
        target = safe_repo_file_path(lmstudio_repo_dir(Path(base_model_dir), repo_id), filename)
    except HfRepoError:
        return {}
    part = target.with_suffix(target.suffix + ".part")
    try:
        if not part.exists():
            return {}
        partial_size = part.stat().st_size
    except OSError:
        return {}
    return {
        "partial_path": str(part),
        "partial_size": partial_size,
        "partial_size_text": format_bytes(partial_size),
        "expected_size": int(expected_size or 0),
        "expected_size_text": format_bytes(expected_size) if expected_size else "—",
    }


def safe_repo_file_path(root: Path, filename: str) -> Path:
    rel = PurePosixPath(filename)
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise HfRepoError(f"Недопустимое имя файла в репозитории: {filename}")
    target = (root / Path(*rel.parts)).resolve()
    root_resolved = root.resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise HfRepoError(f"Недопустимый путь файла: {filename}") from exc
    return target


def resolve_file_url(repo_id: str, filename: str) -> str:
    return HF_RESOLVE_URL.format(
        repo_id=urllib.parse.quote(repo_id, safe="/"),
        filename=urllib.parse.quote(filename, safe="/"),
    )


def fetch_remote_size(repo_id: str, filename: str) -> int:
    """Пробует получить размер файла через HEAD, если API не вернул size."""
    request = urllib.request.Request(
        resolve_file_url(repo_id, filename),
        headers={"User-Agent": USER_AGENT},
        method="HEAD",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return int(response.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def delete_file_safely(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def format_bytes(size: float | int) -> str:
    """Форматирует HF-размеры в decimal units, как на huggingface.co.

    Hugging Face показывает 1 GB = 1_000_000_000 bytes. Если делить на 1024,
    пользователь видит GiB: например 21.4 GB на сайте превращается в 19.9 GiB.
    Поэтому для HF downloader используем decimal GB/TB и подписи без вопросительных
    знаков.
    """
    if not size:
        return "—"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1000
    return str(size)


def format_eta(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
