"""Тесты перечисления локальных моделей (list_all_local_model_entries)."""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.services.hf_downloader import list_all_local_model_entries


class TestListLocalModelEntries(unittest.TestCase):
    def _make_gguf(self, path: Path, size: int = 1024) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00" * size)

    def test_multiple_quants_in_folder_are_separate_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            folder = root / "Qwen3.8-27B-GGUF"
            self._make_gguf(folder / "Qwen3.8-27B-Q4_K_M.gguf")
            self._make_gguf(folder / "Qwen3.8-27B-Q8_0.gguf")
            info = list_all_local_model_entries(root)
            entries = info["entries"]
            self.assertEqual(len(entries), 2)
            names = {e["name"] for e in entries}
            self.assertEqual(
                names,
                {"Qwen3.8-27B-Q4_K_M.gguf", "Qwen3.8-27B-Q8_0.gguf"},
            )
            for e in entries:
                self.assertEqual(e["type"], "file")
                self.assertEqual(e["gguf_count"], 1)

    def test_single_model_folder_is_one_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            folder = root / "Qwen3.8-27B-GGUF"
            self._make_gguf(folder / "Qwen3.8-27B-Q4_K_M.gguf")
            # projector должен игнорироваться как самостоятельная модель
            self._make_gguf(folder / "mmproj-Qwen3.8-27B-f16.gguf")
            info = list_all_local_model_entries(root)
            entries = info["entries"]
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["type"], "folder")
            self.assertEqual(entries[0]["gguf_count"], 1)

    def test_ggufs_directly_in_root_are_separate_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            self._make_gguf(root / "A-Q4.gguf")
            self._make_gguf(root / "B-Q4.gguf")
            info = list_all_local_model_entries(root)
            self.assertEqual(len(info["entries"]), 2)

    def test_split_model_in_folder_is_one_entry_with_full_size(self):
        # e.g. Qwen3.8-Flash-Next-UD-Q2_K_XL, 3-shard split GGUF.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            folder = root / "Flash-Next-GGUF"
            self._make_gguf(folder / "flash-next-Q2_K_XL-00001-of-00003.gguf", size=1000)
            self._make_gguf(folder / "flash-next-Q2_K_XL-00002-of-00003.gguf", size=2000)
            self._make_gguf(folder / "flash-next-Q2_K_XL-00003-of-00003.gguf", size=3000)
            info = list_all_local_model_entries(root)
            entries = info["entries"]
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["type"], "folder")
            self.assertEqual(entries[0]["gguf_count"], 1)
            self.assertEqual(entries[0]["size"], 6000)

    def test_split_model_in_root_is_one_entry_with_full_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            self._make_gguf(root / "flash-next-Q2_K_XL-00001-of-00002.gguf", size=1000)
            self._make_gguf(root / "flash-next-Q2_K_XL-00002-of-00002.gguf", size=1500)
            info = list_all_local_model_entries(root)
            entries = info["entries"]
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["type"], "file")
            self.assertEqual(entries[0]["name"], "flash-next-Q2_K_XL-00001-of-00002.gguf")
            self.assertEqual(entries[0]["size"], 2500)

    def test_split_model_alongside_another_quant_are_two_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            folder = root / "Flash-Next-GGUF"
            self._make_gguf(folder / "flash-next-Q2_K_XL-00001-of-00002.gguf", size=1000)
            self._make_gguf(folder / "flash-next-Q2_K_XL-00002-of-00002.gguf", size=1500)
            self._make_gguf(folder / "flash-next-Q8_0.gguf", size=4000)
            info = list_all_local_model_entries(root)
            entries = info["entries"]
            self.assertEqual(len(entries), 2)
            for e in entries:
                self.assertEqual(e["type"], "file")
                self.assertEqual(e["gguf_count"], 1)
            sizes = {e["name"]: e["size"] for e in entries}
            self.assertEqual(sizes["flash-next-Q2_K_XL-00001-of-00002.gguf"], 2500)
            self.assertEqual(sizes["flash-next-Q8_0.gguf"], 4000)


if __name__ == "__main__":
    unittest.main()
