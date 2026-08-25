from __future__ import annotations

_BASE = (
    "Modern processors execute instructions through pipelines while cache hierarchies reduce effective memory latency. "
    "A cache line has a tag, index and offset; set associativity balances lookup complexity and conflict misses. "
    "Multicore systems require coherence, memory ordering, prefetching and careful synchronization. "
    "Software performance depends on locality, data layout, branching, allocation, contention and working-set size. "
)

_CODE = (
    "Implement and review a production-quality Kotlin LRU cache. Use generic key and value types, explicit capacity, "
    "thread safety, deterministic eviction, clear error handling, documentation and tests. Discuss complexity and "
    "identify race conditions or API design issues. Continue with concrete code and detailed technical reasoning. "
)

_STABILITY_WORKLOADS: tuple[tuple[str, str], ...] = (
    (
        "reasoning",
        "Analyze a subtle concurrency bug in a Kotlin coroutine pipeline. Consider cancellation, exception propagation, "
        "shared mutable state, Flow backpressure, and lifecycle ownership. Compare several plausible explanations before "
        "choosing the most likely root cause. Continue reasoning in detail and do not conclude early.",
    ),
    (
        "code",
        "Write and explain a production-quality Kotlin implementation of a bounded asynchronous work queue. Include "
        "interfaces, coroutine-safe synchronization, cancellation handling, unit-test examples, and a short review of "
        "failure modes. Produce substantial concrete code rather than only prose.",
    ),
    (
        "tool_like",
        "You are an engineering agent inspecting a repository. Produce a sequence of concise investigation notes: identify "
        "files to inspect, commands you would run, observations from hypothetical outputs, then propose a patch and tests. "
        "Use structured steps and realistic command/file names without external tools actually being available.",
    ),
    (
        "structured",
        "Create a detailed technical comparison table in plain text for four cache eviction policies, then derive selection "
        "rules, edge cases, and a deterministic decision procedure. Keep the format regular, repetitive, and explicit so "
        "that many adjacent tokens are structurally predictable.",
    ),
    (
        "mixed",
        "Review an Android architecture where Compose UI, a ViewModel, StateFlow, repository caching, and network retries "
        "interact. Alternate between explanation, pseudo-code, short Kotlin snippets, and checklist-style findings. Discuss "
        "tradeoffs and continue until the design is thoroughly covered.",
    ),
    (
        "narrative_technical",
        "Explain step by step how a request travels through a modern web service from DNS lookup to TLS, load balancing, "
        "application execution, database access, caching, observability, and response delivery. Use connected prose with "
        "occasional examples and continue in depth.",
    ),
    (
        "repair",
        "A large codebase has an intermittent test failure after a refactor. Think through likely causes involving time, "
        "ordering, mocks, shared state, dependency injection, and asynchronous cleanup. Produce a debugging diary with "
        "hypotheses, falsification steps, and a final repair plan.",
    ),
)


def approx_prompt(target_tokens: int, coding: bool = False) -> str:
    # English prose averages roughly 1.3-1.6 tokens/word in modern LLM tokenizers.
    # We intentionally overproduce and use the server-reported token count as ground truth.
    chunk = _CODE if coding else _BASE
    approx_words = max(64, int(target_tokens / 1.4))
    words_per_chunk = len(chunk.split())
    return (chunk + "\n") * max(1, approx_words // words_per_chunk)


def stability_workloads(mode: str = "normal") -> list[tuple[str, str]]:
    """Return deterministic heterogeneous decode workloads.

    quick: 3 samples, normal: 3, deep: 7. NORMAL is a confirm stage, not an exhaustive
    benchmark suite; DEEP retains the broader robustness sweep.
    """
    count = {"quick": 3, "normal": 3, "deep": 7}.get(mode, 3)
    return list(_STABILITY_WORKLOADS[:count])


def context_staircase_prompts(context_size: int, mode: str = "normal") -> list[tuple[int, str]]:
    """Create prompts with a shared prefix so cache_prompt can grow one slot without server restart."""
    if mode == "quick":
        fractions = (0.125, 0.50)
    elif mode == "normal":
        fractions = (0.125, 0.50, 0.75)
    else:
        fractions = (0.125, 0.25, 0.50, 0.75)
    # Keep enough room for generated tokens and template/tokenizer variation.
    max_target = max(1024, int(context_size * 0.78))
    targets: list[int] = []
    for f in fractions:
        target = min(max_target, max(1024, int(context_size * f)))
        if not targets or target > targets[-1]:
            targets.append(target)
    prefix = (
        "CONTEXT STAIRCASE BENCHMARK. Preserve this prefix exactly across stages. "
        "The following technical notebook discusses CPU caches, concurrency, networking, databases, Android architecture, "
        "and software debugging. Read the entire accumulated notebook, then continue with a concise technical synthesis.\n\n"
    )
    rows: list[tuple[int, str]] = []
    for target in targets:
        body = approx_prompt(max(512, target - 128), coding=False)
        rows.append((target, prefix + body))
    return rows
