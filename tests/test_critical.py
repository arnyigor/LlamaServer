"""Тесты для критичных частей LlamaServer GUI."""

import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.constants import LLAMA_ALLOWED_FLAGS
from src.core.mem_viz_parser import MemoryData, parse_line
from src.core.gguf_parser import (
    extract_model_info,
    recommend_context,
    quant_from_filename,
    detect_mtp_draft_for_model,
    is_mtp_draft_file,
    is_projector_file,
)
from src.utils.file_utils import (
    validate_path,
    write_json_file_safely,
    load_or_create_json,
)
from src.services.hf_downloader import (
    list_all_local_model_entries,
    list_all_partial_downloads,
)
from src.services.threads import LlamaCppUpdater


class TestValidatePath(unittest.TestCase):
    """Тесты валидации путей."""

    def test_valid_path(self):
        """Валидный путь должен проходить."""
        result = validate_path("/tmp/test")
        self.assertIsInstance(result, Path)

    def test_must_exist(self):
        """Проверка существования файла."""
        with self.assertRaises(ValueError):
            validate_path("/nonexistent/path", must_exist=True)

    def test_base_dir_validation(self):
        """Путь должен быть внутри base_dir."""
        base = Path("/tmp")
        # Внутри base_dir — OK
        result = validate_path("/tmp/subdir/file", base_dir=base)
        self.assertEqual(result, Path("/tmp/subdir/file").resolve())

    def test_base_dir_escape(self):
        """Путь за пределами base_dir должен отклоняться."""
        base = Path("/tmp")
        with self.assertRaises(ValueError):
            validate_path("/etc/passwd", base_dir=base)


class TestHfPartialDiscovery(unittest.TestCase):
    def test_lists_partial_downloads_from_all_lmstudio_repositories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "author-one" / "model-a" / "model-Q4.gguf.part"
            second = root / "author-two" / "model-b" / "nested" / "model-Q8.gguf.part"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_bytes(b"a" * 1024)
            second.write_bytes(b"b" * 2048)

            partials = list_all_partial_downloads(root)

        self.assertEqual(len(partials), 2)
        self.assertEqual(partials[0]["repo_id"], "author-one/model-a")
        self.assertEqual(partials[0]["rfilename"], "model-Q4.gguf")
        self.assertEqual(partials[0]["partial_size"], 1024)
        self.assertEqual(partials[1]["repo_id"], "author-two/model-b")
        self.assertEqual(partials[1]["rfilename"], "nested/model-Q8.gguf")


class TestGGUFParser(unittest.TestCase):
    """Тесты парсера GGUF."""

    def test_quant_from_filename(self):
        """Определение квантования из имени файла."""
        test_cases = [
            ("model-Q4_K_M.gguf", "Q4_K_M"),
            ("model-Q8_0.gguf", "Q8_0"),
            ("model-F16.gguf", "F16"),
            ("model-IQ4_XS.gguf", "IQ4_XS"),
            ("model.gguf", ""),
        ]
        for filename, expected in test_cases:
            with self.subTest(filename=filename):
                result = quant_from_filename(filename)
                self.assertEqual(result, expected)

    def test_is_projector_file(self):
        """Определение projector-файлов."""
        self.assertTrue(is_projector_file("mmproj-model.gguf"))
        self.assertTrue(is_projector_file("model-projector.gguf"))
        self.assertFalse(is_projector_file("model-Q4_K_M.gguf"))

    def test_mtp_draft_detection(self):
        """Поиск отдельного MTP draft GGUF в пакете модели."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model = root / "gemma-4-26B-Q4_K_M.gguf"
            model.write_bytes(b"fake")
            draft_dir = root / "MTP"
            draft_dir.mkdir()
            draft = draft_dir / "gemma-4-26B-MTP-draft-Q8_0.gguf"
            draft.write_bytes(b"fake")

            self.assertTrue(is_mtp_draft_file(draft))
            self.assertEqual(detect_mtp_draft_for_model(model), str(draft))

    def test_mtp_draft_detection_ignores_sibling_model_folders(self):
        """MTP draft search must not cross-pair LM Studio sibling models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            author = Path(tmpdir) / "unsloth"
            gemma_dir = author / "gemma-4-26B-A4B-it-qat-GGUF"
            qwen_dir = author / "Qwen3.6-27B-MTP-GGUF"
            gemma_dir.mkdir(parents=True)
            qwen_dir.mkdir(parents=True)
            model = gemma_dir / "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
            model.write_bytes(b"fake")
            qwen_draft = qwen_dir / "Qwen3.6-27B-UD-IQ3_XXS.gguf"
            qwen_draft.write_bytes(b"fake")

            self.assertEqual(detect_mtp_draft_for_model(model), "")

    def test_mtp_draft_detection_ignores_other_quant_in_same_mtp_package(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "Qwen3.6-27B-MTP-GGUF"
            root.mkdir(parents=True)
            model = root / "Qwen3.6-27B-UD-Q3_K_XL.gguf"
            other_quant = root / "Qwen3.6-27B-UD-IQ3_XXS.gguf"
            model.write_bytes(b"fake")
            other_quant.write_bytes(b"fake")

            self.assertFalse(is_mtp_draft_file(other_quant))
            self.assertEqual(detect_mtp_draft_for_model(model), "")

    def test_list_all_local_model_entries(self):
        """Local manager lists all model folders, not only downloaded HF repos."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            lm_model = root / "unsloth" / "model-a"
            lm_model.mkdir(parents=True)
            (lm_model / "model-a-Q4_K_M.gguf").write_bytes(b"a" * 10)
            (lm_model / "mmproj-model-a.gguf").write_bytes(b"p" * 5)
            direct = root / "direct-model-Q8_0.gguf"
            direct.write_bytes(b"d" * 7)

            result = list_all_local_model_entries(root)
            relatives = {
                str(entry["relative"]).replace("\\", "/") for entry in result["entries"]
            }

            self.assertIn("unsloth/model-a", relatives)
            self.assertIn("direct-model-Q8_0.gguf", relatives)
            self.assertEqual(len(result["entries"]), 2)

    def test_recommend_context(self):
        """Рекомендация размера контекста."""
        # Маленькая модель Q2 (3 GiB <= 5 threshold)
        # Q2 -> recommended 4096, но size <= 5 и quant есть -> max(4096, 8192) = 8192
        # model_ctx = 8192 -> min(8192, 8192) = 8192
        info = {"quant": "Q2_K", "size_gib": 3, "context_length": 8192}
        result = recommend_context(info)
        self.assertEqual(result, 8192)

        # Большая модель F16
        info = {"quant": "F16", "size_gib": 30, "context_length": 32768}
        result = recommend_context(info)
        self.assertEqual(result, 8192)  # Ограничено размером модели

        # Средняя модель Q4
        info = {"quant": "Q4_K_M", "size_gib": 8, "context_length": 32768}
        result = recommend_context(info)
        # Q4 -> 8192, size 8 < 14 (MEDIUM) и > 5 (SMALL), нет ограничений
        self.assertEqual(result, 8192)


class TestLlamaCppUpdater(unittest.TestCase):
    """Тесты обновления llama.cpp."""

    def setUp(self):
        """Подготовка тестов."""
        self.updater = LlamaCppUpdater("/fake/llama-server.exe")

    def test_parse_build_number(self):
        """Парсинг номера билда из тега."""
        test_cases = [
            ("b4000", 4000),
            ("b1234", 1234),
            ("v1.0", None),
            ("", None),
            ("b999", 999),  # Маленький номер тоже валиден
            ("b1000", 1000),
            ("4600", 4600),  # Без префикса b
        ]
        for tag, expected in test_cases:
            with self.subTest(tag=tag):
                result = self.updater.parse_build_number(tag)
                self.assertEqual(result, expected)

    def test_select_assets_cuda(self):
        """Выбор CUDA ассета."""
        release = {
            "assets": [
                {"name": "llama-b4000-bin-win-cuda-12.4-x64.zip"},
                {"name": "llama-b4000-bin-win-vulkan-x64.zip"},
            ]
        }
        result = self.updater.select_assets(release)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "llama-b4000-bin-win-cuda-12.4-x64.zip")

    def test_select_assets_fallback(self):
        """Fallback на любой Windows билд."""
        release = {
            "assets": [
                {"name": "llama-b4000-bin-win-vulkan-x64.zip"},
            ]
        }
        result = self.updater.select_assets(release)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "llama-b4000-bin-win-vulkan-x64.zip")

    def test_select_assets_empty(self):
        """Пустой список ассетов."""
        release = {"assets": []}
        result = self.updater.select_assets(release)
        self.assertEqual(len(result), 0)

    @patch("urllib.request.urlopen")
    def test_fetch_latest_release(self, mock_urlopen):
        """Получение последнего релиза."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {"tag_name": "b4000", "assets": [{"name": "test.zip"}]}
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        result = self.updater.fetch_latest_release()
        self.assertEqual(result["tag_name"], "b4000")

    @patch("urllib.request.urlopen")
    def test_fetch_latest_release_skips_marker(self, mock_urlopen):
        """Маркер-релиз (v0.2.0) без ассетов пропускается, берётся b-релиз."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            [
                {"tag_name": "v0.2.0", "assets": []},
                {
                    "tag_name": "b10615",
                    "assets": [
                        {"name": "llama-b10615-bin-win-cuda-12.4-x64.zip"},
                    ],
                },
            ]
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        result = self.updater.fetch_latest_release()
        self.assertEqual(result["tag_name"], "b10615")

    def test_safe_extract_zip(self):
        """Безопасная распаковка ZIP."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Создаем тестовый ZIP
            zip_path = tmpdir / "test.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("llama-server.exe", "fake exe content")
                zf.writestr("subdir/file.dll", "fake dll content")

            # Распаковываем
            extract_dir = tmpdir / "extract"
            self.updater.safe_extract_zip(zip_path, extract_dir)

            # Проверяем
            self.assertTrue((extract_dir / "llama-server.exe").exists())
            self.assertTrue((extract_dir / "subdir" / "file.dll").exists())

    def test_safe_extract_zip_traversal(self):
        """Защита от path traversal в ZIP."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Создаем вредоносный ZIP
            zip_path = tmpdir / "evil.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("../../../etc/passwd", "evil content")

            # Должен выбросить исключение
            with self.assertRaises(RuntimeError):
                self.updater.safe_extract_zip(zip_path, tmpdir / "extract")


class TestMemoryVizParser(unittest.TestCase):
    """Тесты парсера вкладки Memory."""

    def test_unload_clears_allocations(self):
        data = MemoryData()
        parse_line("CUDA0 model buffer size = 4096.00 MiB", data)
        self.assertGreater(data.grand_total(), 0)

        parse_line("llama_model_unload: model unloaded", data)

        self.assertEqual(data.grand_total(), 0)
        self.assertFalse(data.server_ready)


class TestConstants(unittest.TestCase):
    """Тесты констант."""

    def test_allowed_flags_not_empty(self):
        """Whitelist не должен быть пустым."""
        self.assertGreater(len(LLAMA_ALLOWED_FLAGS), 0)

    def test_allowed_flags_format(self):
        """Все флаги должны начинаться с --."""
        for flag in LLAMA_ALLOWED_FLAGS:
            self.assertTrue(flag.startswith("--"), f"Flag {flag} doesn't start with --")


class TestFileUtils(unittest.TestCase):
    """Тесты утилит файловой системы."""

    def test_write_json_file_safely(self):
        """Атомарная запись JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {"key": "value", "number": 42}

            write_json_file_safely(path, data)

            # Проверяем, что файл создан
            self.assertTrue(path.exists())

            # Проверяем содержимое
            with open(path, "r") as f:
                loaded = json.load(f)
            self.assertEqual(loaded, data)

    def test_load_or_create_json_existing(self):
        """Загрузка существующего JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {"key": "value"}
            with open(path, "w") as f:
                json.dump(data, f)

            result = load_or_create_json(path)
            self.assertEqual(result, data)

    def test_load_or_create_json_new(self):
        """Создание нового JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "new.json"
            result = load_or_create_json(path)
            self.assertEqual(result, {})


class TestIntegration(unittest.TestCase):
    """Тесты интеграции."""

    def test_build_args_import(self):
        """Проверка импорта build_args."""
        from src.core.cli_builder import build_args

        self.assertTrue(callable(build_args))


def run_tests():
    """Запуск всех тестов."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Добавляем все тесты
    suite.addTests(loader.loadTestsFromTestCase(TestValidatePath))
    suite.addTests(loader.loadTestsFromTestCase(TestGGUFParser))
    suite.addTests(loader.loadTestsFromTestCase(TestLlamaCppUpdater))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryVizParser))
    suite.addTests(loader.loadTestsFromTestCase(TestConstants))
    suite.addTests(loader.loadTestsFromTestCase(TestFileUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
