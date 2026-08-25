"""Phase 1 structural verification: instantiate MainWindowUI headless,
assert all attributes referenced by main.py exist, exercise nav pages."""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.ui.main_window import MainWindowUI

REQUIRED = [
    # header / nav / layout
    "header",
    "language_combo",
    "status_bar_widget",
    "launch_controls_widget",
    "nav_rail",
    "pages",
    "content_splitter",
    "main_vsplit",
    "log_dock",
    # overview / dashboard
    "overview_status",
    "overview_model",
    "overview_speed_value",
    "overview_vram_value",
    "overview_request_value",
    "overview_context_value",
    "overview_active_value",
    "overview_endpoint_value",
    "overview_settings",
    "overview_memory_note",
    "overview_content_widget",
    "model_group",
    "runtime_stats_group",
    "launch_summary_group",
    # status bar
    "status_indicator",
    "status_short",
    "status_speed",
    "status_vram",
    # preflight
    "preflight_status",
    "preflight_model",
    "preflight_context",
    "preflight_kv",
    "preflight_gpu",
    "preflight_mtp",
    "preflight_endpoint",
    "preflight_warning",
    # logs
    "logs",
    "autoscroll_logs",
    "copy_last_error_btn",
    "open_diagnostics_btn",
    # pages content
    "paths_panel",
    "g_launch",
    "adv_panel",
    "sampling_panel",
    "server_panel",
    "models_panel",
    "int_panel",
    "bench_panel",
    "cli_group",
    "autotune",
    # a sample of param widgets used by main.py
    "start_btn",
    "stop_btn",
    "reload_btn",
    "force_stop_btn",
    "advanced_mode_chk",
    "model_combo",
    "scan_btn",
    "ctx_size",
    "gpu_layers",
    "gpu_auto",
    "temperature",
    "top_k",
    "host",
    "port",
    "extra_args",
    "cli_preview",
    "test_btn",
    "hf_repo",
    "integration_target",
    "preset_name_combo",
    "speculative_mtp",
    "spec_draft_n_max",
    "update_llama_btn",
    "exe_path",
    "model_dir",
    "cuda_version_combo",
]


def main():
    app = QApplication(sys.argv)
    ui = MainWindowUI()

    missing = [a for a in REQUIRED if not hasattr(ui, a)]
    if missing:
        print("FAIL: missing attributes:", missing)
        sys.exit(1)
    print(f"OK: all {len(REQUIRED)} required attributes present")

    # Exercise every nav page (must not raise)
    n = ui.pages.count()
    print(f"OK: {n} nav pages")
    for i in range(n):
        ui.nav_rail.setCurrentRow(i)
        assert ui.pages.currentIndex() == i, f"page index mismatch at {i}"
    print("OK: nav page switching works")

    # Advanced mode toggle must not raise
    ui.advanced_mode_chk.setChecked(False)
    ui.advanced_mode_chk.setChecked(True)
    print("OK: advanced mode toggle works")

    # Save/restore UI state round-trip
    ui.save_ui_state()
    ui._load_ui_state()
    print("OK: save/load ui state round-trip")

    ui.close()
    print("PHASE 1 STRUCTURAL VERIFICATION PASSED")


if __name__ == "__main__":
    main()
