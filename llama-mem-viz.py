#!/usr/bin/env python3
"""
llama-mem-viz.py — stdlib-only memory visualization for llama.cpp (RAM / VRAM),
accounting for out-of-memory errors (OOM / failed to allocate / cudaMalloc failed).

Examples:
  python llama-mem-viz.py -m model.gguf -c 65536 -ngl 40
  python llama-mem-viz.py -m model.gguf -c 65536 -ngl 40 --save-log llama.log
  python llama-mem-viz.py -m model.gguf -c 65536 -ngl 40 --show-log
  python llama-mem-viz.py --log llama.log
  python llama-mem-viz.py --log llama.log --json
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from src.core.mem_viz_parser import (
    COMPONENT_META,
    COMPONENT_ORDER,
    MemoryData,
    estimate_shortfall_mib,
    fmt_mem,
    parse_line,
    should_stop,
)


# ─────────────────────────────────────────────────────────────
# ANSI Colors
# ─────────────────────────────────────────────────────────────

USE_COLOR = sys.stdout.isatty()

RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"


def c(code: str) -> str:
    return code if USE_COLOR else ""


def fg(n: int) -> str:
    return c(f"\033[38;5;{n}m")


def bg(n: int) -> str:
    return c(f"\033[48;5;{n}m")


def strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def get_ram_stats_mib() -> tuple[float | None, float | None]:
    total = None
    available = None
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) / 1024.0
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) / 1024.0
                if total is not None and available is not None:
                    break
    except Exception:
        pass
    return total, available


def get_nvidia_vram_stats_mib() -> tuple[float | None, float | None]:
    total = 0.0
    free = 0.0
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in out.splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                total += float(parts[0])
                free += float(parts[1])
    except Exception:
        return None, None

    return (total if total > 0 else None, free if free > 0 else None)


def get_system_memory() -> dict[str, float]:
    mem: dict[str, float] = {}

    ram_total, ram_avail = get_ram_stats_mib()
    if ram_total is not None:
        mem["RAM"] = ram_total
    if ram_avail is not None:
        mem["RAM_FREE"] = ram_avail

    vram_total, vram_free = get_nvidia_vram_stats_mib()
    if vram_total is not None:
        mem["VRAM"] = vram_total
    if vram_free is not None:
        mem["VRAM_FREE"] = vram_free

    return mem


# ─────────────────────────────────────────────────────────────
# Log Sources
# ─────────────────────────────────────────────────────────────

BINARY_CANDIDATES = [
    "llama-server", "llama-cli",
    "./llama-server", "./llama-cli",
    "llama.cpp/build/bin/llama-server",
    "llama.cpp/build/bin/llama-cli",
]


def find_binary() -> str:
    for b in BINARY_CANDIDATES:
        candidate = os.path.expanduser(b)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    raise FileNotFoundError("Could not find llama-server or llama-cli binary.")


def collect_from_process(
        binary: str,
        llama_args: list[str],
        show_log: bool = False,
        save_log: Path | None = None,
        debug: bool = False,
) -> MemoryData:
    data = MemoryData()
    data.system_memory = get_system_memory()

    binary = os.path.expanduser(binary)
    base   = os.path.basename(binary)

    if "llama-cli" in base:
        if "-p" not in llama_args:
            llama_args = llama_args + ["-p", "hi", "-n", "1"]

    cmd = [binary] + llama_args

    if show_log:
        print(f"{c(BOLD)}▶  {' '.join(cmd)}{c(RESET)}", file=sys.stderr)
        print("─" * shutil.get_terminal_size((120, 40)).columns, file=sys.stderr)

    log_f = None
    if save_log is not None:
        log_f = open(save_log, "w", encoding="utf-8", errors="replace")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        errors="replace",
    )

    try:
        for line in proc.stdout:
            if log_f is not None:
                log_f.write(line)
            if show_log:
                sys.stderr.write(line)
                sys.stderr.flush()

            parse_line(line, data, debug=debug)

            # Stop scanning when server initiates/completes startup sequence
            if should_stop(line):
                proc.terminate()
                break
    finally:
        if log_f is not None:
            log_f.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

        data.process_exit_code = proc.returncode

    return data


def collect_from_file(path: Path, debug: bool = False) -> MemoryData:
    data = MemoryData()
    data.system_memory = get_system_memory()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parse_line(line, data, debug=debug)
    return data


def collect_from_stdin(debug: bool = False) -> MemoryData:
    data = MemoryData()
    data.system_memory = get_system_memory()
    for line in sys.stdin:
        parse_line(line, data, debug=debug)
    return data


# ─────────────────────────────────────────────────────────────
# Visualization Layout
# ─────────────────────────────────────────────────────────────

def allocate_widths(values: list[float], width: int) -> list[int]:
    total = sum(values)
    if total <= 0:
        return [0] * len(values)

    raw    = [v / total * width for v in values]
    floors = [max(1, int(r)) if v > 0 else 0 for r, v in zip(raw, values)]

    diff = width - sum(floors)
    if diff > 0:
        idxs = sorted(range(len(values)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
        for i in idxs[:diff]:
            floors[i] += 1
    elif diff < 0:
        for _ in range(-diff):
            cands = [i for i, v in enumerate(values) if v > 0 and floors[i] > 1]
            if not cands:
                break
            i = max(cands, key=lambda j: floors[j])
            floors[i] -= 1

    return floors


def render_stacked_bar(comps: dict[str, float], components: list[str], width: int) -> str:
    values = [comps.get(comp, 0.0) for comp in components]
    widths = allocate_widths(values, width)

    out = []
    for comp, w in zip(components, widths):
        if w <= 0:
            continue
        meta = COMPONENT_META[comp]
        out.append(f"{bg(meta['color'])}{fg(meta['color'])}{meta['char'] * w}{c(RESET)}")

    visible = sum(widths)
    if visible < width:
        out.append(f"{c(DIM)}{'·' * (width - visible)}{c(RESET)}")

    return "".join(out)


def render_util_bar(util: float, width: int) -> str:
    util   = max(0.0, min(100.0, util))
    filled = round(width * util / 100.0)

    if util >= 90:
        color = 196
    elif util >= 75:
        color = 208
    else:
        color = 71

    return f"{fg(color)}{'━' * filled}{c(RESET)}{c(DIM)}{'╌' * (width - filled)}{c(RESET)}"


def print_model_header(data: MemoryData) -> None:
    info = data.model_info
    if not info:
        return

    parts = []
    if "name" in info:
        parts.append(f"{c(BOLD)}{info['name']}{c(RESET)}")

    for key, label in [
        ("arch", "arch"),
        ("model_type", "type"),
        ("file_format", "format"),
        ("file_type", "quant"),
        ("params", "params"),
        ("file_size", "file"),
        ("ctx", "ctx"),
        ("n_seq_max", "seqs"),
    ]:
        if key in info:
            parts.append(f"{label}: {info[key]}")

    if "layers_offloaded" in info and "layers_total" in info:
        parts.append(f"offload: {info['layers_offloaded']}/{info['layers_total']} layers")

    print()
    print("  " + " │ ".join(parts))


def print_status_banner(
        data: MemoryData,
        failure_cat: str | None = None,
        remaining_mib: float | None = None,
        deficit_mib: float | None = None,
) -> None:
    if not data.fatal_error and not data.warnings:
        return

    print()

    if data.fatal_error:
        print(f"{bg(160)}{fg(15)}  ⚠  SERVER OUT OF MEMORY / LAUNCH FAILED  {c(RESET)}")
        print(f"  {fg(196)}Reason:{c(RESET)} {data.fatal_error}")

        if data.failed_component:
            print(f"  {fg(196)}Affected Component:{c(RESET)} {COMPONENT_META[data.failed_component]['label']}")

        if data.failed_alloc_mib is not None:
            print(f"  {fg(196)}Last Allocation Request:{c(RESET)} {fmt_mem(data.failed_alloc_mib)}")

        if failure_cat is not None and remaining_mib is not None:
            print(f"  {fg(196)}Estimated Remaining {failure_cat}:{c(RESET)} {fmt_mem(remaining_mib)}")

        if deficit_mib is not None and deficit_mib > 0:
            print(f"  {fg(196)}Estimated Deficit:{c(RESET)} {fmt_mem(deficit_mib)}")

        if data.failure_lines:
            extras = [x for x in data.failure_lines if x != data.fatal_error][:2]
            for extra in extras:
                print(f"  {c(DIM)}↳ {extra}{c(RESET)}")

        if data.server_ready:
            print(f"  {c(DIM)}The server started successfully but crashed afterwards.{c(RESET)}")

    elif data.warnings:
        print(f"{bg(208)}{fg(0)}  WARNINGS  {c(RESET)}")

    for w in data.warnings[-3:]:
        print(f"  {fg(208)}⚠ {w}{c(RESET)}")


def visualize(data: MemoryData) -> None:
    agg = data.get_aggregated()
    failure_cat, remaining_mib, deficit_mib = estimate_shortfall_mib(data)

    if not any(agg.values()):
        print_status_banner(data, failure_cat, remaining_mib, deficit_mib)
        if not (data.fatal_error or data.warnings):
            print(f"\n{c(BOLD)}⚠  No memory data found.{c(RESET)}")
        return

    term_width = max(80, shutil.get_terminal_size((120, 40)).columns)
    sep        = "─" * term_width
    components = data.components_used()

    label_w = max(len(COMPONENT_META[comp]["label"]) for comp in components) + 2
    bar_w   = max(24, term_width - label_w - 34)

    # Header Panel
    print()
    print(f"{c(BOLD)}{'═' * term_width}{c(RESET)}")
    print(f"{c(BOLD)}{'llama.cpp memory':^{term_width}}{c(RESET)}")
    print(f"{c(BOLD)}{'═' * term_width}{c(RESET)}")

    print_model_header(data)
    print_status_banner(data, failure_cat, remaining_mib, deficit_mib)

    # Legend Panel
    print()
    print(sep)
    legend = "   ".join(
        f"{fg(COMPONENT_META[comp]['color'])}{COMPONENT_META[comp]['char']}{COMPONENT_META[comp]['char']}{c(RESET)} {COMPONENT_META[comp]['label']}"
        for comp in components
    )
    print("  " + legend)
    print(sep)

    for cat in ["VRAM", "RAM"]:
        comps = agg.get(cat, {})
        total = sum(comps.values())
        if total <= 0:
            continue

        util = data.utilization(cat)
        cap  = data.system_memory.get(cat)

        util_s = ""
        if util is not None and cap is not None:
            util_s = f"  {c(DIM)}{util:.1f}% of {fmt_mem(cap, short=True)}{c(RESET)}"

        if data.fatal_error and failure_cat == cat:
            util_s += f"  {fg(196)}⛔ OOM{c(RESET)}"

        if data.fatal_error and data.system_memory.get(f"{cat}_FREE") is not None:
            util_s += f"  {c(DIM)}free: {fmt_mem(data.system_memory[f'{cat}_FREE'], short=True)}{c(RESET)}"

        print()
        print(f"  {c(BOLD)}{cat}{c(RESET)} — {fmt_mem(total)}{util_s}")

        if util is not None:
            print(f"  {render_util_bar(util, bar_w + label_w)} {util:.1f}%")

        print(f"  {'─' * (term_width - 4)}")

        stacked = render_stacked_bar(comps, components, bar_w)
        label_total = "Total".ljust(label_w)
        print(f"  {c(BOLD)}{label_total}{c(RESET)}┤{stacked}│ {c(BOLD)}{fmt_mem(total):>10} 100.0%{c(RESET)}")

        for comp in components:
            mib = comps.get(comp, 0.0)
            if mib <= 0:
                continue

            meta  = COMPONENT_META[comp]
            pct   = mib / total * 100 if total else 0.0
            one   = render_stacked_bar({comp: mib}, [comp], bar_w)
            label = meta["label"].ljust(label_w)

            print(
                f"  {fg(meta['color'])}{label}{c(RESET)}│{one}│ "
                f"{fmt_mem(mib):>10} {pct:5.1f}%"
            )

    print()
    print(sep)
    print(f"{c(BOLD)}  {'':<6}", end="")
    for comp in components:
        print(f"  {COMPONENT_META[comp]['label']:>12}", end="")
    print(f"  {'TOTAL':>10}  {'Used':>8}{c(RESET)}")

    print(f"  {'─' * 6}", end="")
    for _ in components:
        print(f"  {'─' * 12}", end="")
    print(f"  {'─' * 10}  {'─' * 8}")

    for cat in ["VRAM", "RAM"]:
        comps = agg.get(cat, {})
        total = sum(comps.values())
        util  = data.utilization(cat)

        print(f"  {cat:<6}", end="")
        for comp in components:
            mib = comps.get(comp, 0.0)
            clr = fg(COMPONENT_META[comp]["color"]) if mib > 0 else c(DIM)
            print(f"  {clr}{fmt_mem(mib, short=True):>12}{c(RESET)}", end="")
        util_s = f"{util:7.1f}%" if util is not None else "       —"
        print(f"  {c(BOLD)}{fmt_mem(total, short=True):>10}{c(RESET)}  {util_s}")

    print(f"  {'─' * 6}", end="")
    for _ in components:
        print(f"  {'─' * 12}", end="")
    print(f"  {'─' * 10}  {'─' * 8}")

    print(f"  {c(BOLD)}{'TOTAL':<6}", end="")
    for comp in components:
        comp_total = sum(agg[cat].get(comp, 0.0) for cat in ["VRAM", "RAM"])
        print(f"  {fmt_mem(comp_total, short=True):>12}", end="")
    print(f"  {fmt_mem(sum(sum(agg[cat].values()) for cat in ['VRAM', 'RAM']), short=True):>10}{c(RESET)}")

    print(sep)
    print()


# ─────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="llama.cpp memory visualization (RAM / VRAM) with OOM detection",
    )

    src = p.add_mutually_exclusive_group()
    src.add_argument("--log", type=Path, help="read saved log file")
    src.add_argument("--stdin", action="store_true", help="read from stdin")

    p.add_argument("-b", "--binary", help="path to llama-server or llama-cli binary")
    p.add_argument("--show-log", action="store_true", help="print llama.cpp log to stderr")
    p.add_argument("--save-log", type=Path, help="save llama.cpp log to file")
    p.add_argument("--json", action="store_true", help="output JSON")
    p.add_argument("--debug", action="store_true", help="debug regex matches")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colors")

    return p.parse_known_args()


# В начале main()
def main() -> None:
    global USE_COLOR

    args, llama_args = parse_args()
    if args.no_color:
        USE_COLOR = False

    # Сохраняем полную командную строку
    cli_args = " ".join(llama_args) if llama_args else ""

    if args.log:
        data = collect_from_file(args.log, debug=args.debug)
    elif args.stdin:
        data = collect_from_stdin(debug=args.debug)
    else:
        if not llama_args:
            print(__doc__)
            sys.exit(1)

        binary = args.binary or find_binary()
        data = collect_from_process(
            binary=binary,
            llama_args=llama_args,
            show_log=args.show_log,
            save_log=args.save_log,
            debug=args.debug,
        )

    # Добавляем аргументы в объект
    data.cli_args = cli_args

    if args.json:
        print(json.dumps(data.to_dict(), indent=2, ensure_ascii=False))
    else:
        visualize(data)

if __name__ == "__main__":
    main()
