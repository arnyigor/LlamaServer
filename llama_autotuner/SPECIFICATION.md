# MVP specification and roadmap

This is the single source of truth for the autotuner's algorithm and hard behavioral rules.
Everything here must match the current code; when they disagree, fix the code or fix this file in
the same commit — never let them drift. Project-level narrative (what the tool is, how to install
and run it) lives in [README.md](README.md); release-by-release history lives in
[CHANGELOG.md](CHANGELOG.md). This file does not repeat either.

## Objective

Automatically derive reproducible `llama-server` launch profiles for a specific GGUF + machine by testing the real server workload. Synthetic `llama-bench` data is advisory only.

## v0.4 target semantics

The primary input is now a **user target**, not an implicit fixed benchmark recipe. `TargetSpec` includes requested context, workload, Vision requirement, preferred KV precision, memory reserve, optional minimum PP/TG, and a degradation policy.

Before expensive benchmarking the application must:

1. detect model capabilities;
2. reject impossible required capabilities (for example required Vision on a text-only model);
3. estimate exact-target memory feasibility;
4. classify the machine/model pair as COMFORTABLE, CONSTRAINED, INFEASIBLE or UNKNOWN;
5. generate explicit solution alternatives when the exact preferred-quality target is not attractive;
6. label every change as performance, quality-risk, or capability degradation.

No quality/capability degradation may be applied silently.

## Official v0.1 support boundary

- Windows 10/11.
- One NVIDIA GPU.
- CUDA llama.cpp build.
- GGUF models.
- Dense and MoE architectures.
- CPU/GPU hybrid offload.
- `np=1`.
- MTP when both GGUF and llama.cpp expose it.

The application must reject unsupported autotune environments rather than silently claiming reliable tuning.

## Environmental invariants

Before tuning:

1. sample GPU state multiple times;
2. record stable desktop/browser/YouTube/IDE usage as baseline;
3. advise the user not to start games, renderers, Stable Diffusion, another LLM server or other heavy CUDA jobs;
4. classify environment as CLEAN/MODERATE/BUSY;
5. refuse BUSY state in unattended mode.

Before every candidate:

1. previous server must be stopped;
2. VRAM must return to session baseline within tolerance;
3. a large unexplained baseline change invalidates the next run.

During candidates the tool samples GPU memory so report headroom is based on measured runtime state rather than a static file-size estimate.

## Pipeline

```text
DISCOVER HARDWARE
  -> GPU PREFLIGHT
  -> LLAMA CAPABILITIES
  -> GGUF INSPECTION
  -> TARGET / CAPABILITY / FEASIBILITY PLAN
  -> EXPLICIT SOLUTION OPTION (Solution Envelope, ranked)
  -> SAFE PLACEMENT (Dense: NGL, MoE: NCMOE)
  -> JOINT UBATCH + PLACEMENT SEARCH
  -> CPU THREAD-BATCH A/B
  -> LOCAL PLACEMENT REFINEMENT
  -> MTP SCREEN (auto: reject / speed-only FULL / workload-earned expansion)
  -> MTP N-MAX / P-MIN (DEEP or workload-earned only)
  -> FINAL MTP + NON-MTP CANDIDATE SET
  -> CANDIDATE-LOCAL FINAL FALLBACK (bounded; never on INVALID_ENVIRONMENT)
  -> HETEROGENEOUS DECODE STABILITY
  -> CACHED CONTEXT STAIRCASE / LONG-CONTEXT VALIDATION
  -> ROBUST PROFILE GENERATION (OPTIMAL / MAX_CONTEXT / MAX_KV_PRECISION / FASTEST)
```

**Static analysis only proposes and ranks candidates; the final decision always comes from a real
`llama-server` launch.** Formula-based VRAM/speed estimates carry MEDIUM confidence and can be
contradicted by measurement — observed in practice: a statically-plausible `ngl=64` triggered
`EARLY_REJECT_ABSOLUTE_VRAM_FLOOR` at runtime and the search corrected to `ngl=63`.

## Solution envelope

For a given target the feasibility planner builds several families of alternatives, not one point:

| Family | What it changes | Example |
|---|---|---|
| `EXACT_TARGET` | nothing (exact request) | `ctx=262144, KV=f16/f16` |
| `PRESERVE_CONTEXT_KV_*` | lowers KV precision, keeps context | `ctx=262144, KV=q4_0/q4_0` |
| `PRESERVE_KV_PRECISION_CTX_*` | lowers context, keeps KV | `ctx=16384, KV=f16/f16` |
| `TRADEOFF_CTX_*_KV_*` | lowers both context and KV | `ctx=65536, KV=q4_0/q4_0` |

Each option carries a predicted placement (how many layers fit on GPU), predicted free VRAM, and a
resource class. For Dense models, if pure full-GPU (`ngl=all`) does not clear the absolute VRAM
floor, the option is **not discarded**: a numeric `ngl` (partial CPU offload) is computed instead,
so a near-full-GPU point at a moderate context is not silently thrown away.

Target invariants:

- Requested context is a capability requirement, not merely the first value in a fallback loop.
- Required Vision cannot silently become text-only.
- Lower KV precision is `quality-risk`, not a free optimization.
- Lower context is `capability` degradation and never earns exact-target status.
- Dense target-layer CPU offload is `performance` degradation with a high prior cost.
- MoE routed-expert CPU offload has a different architecture strategy and may be a normal
  first-class fit mechanism, not a degradation of last resort.
- A runnable candidate that misses explicit `min TG/PP` is diagnostic `PASS_DEGRADED`, not a
  recommendation.

Resource-class branches:

- `COMFORTABLE`: preserve target/capabilities/precision and spend the search budget on
  performance (batch/ubatch/MTP). Memory-recovery sweeps are pruned; resource-rich MoE can test
  `ncmoe=0` directly.
- `CONSTRAINED`: preserve requested semantics first, measure headroom, and compare explicit
  trade-offs when useful.
- `INFEASIBLE`: do not pretend a parameter search can solve a capability mismatch. Generate
  explicit KV/context/placement alternatives according to degradation policy.

## Ranking (`sort_key`) — which envelope option is tried first

Dense and MoE are ranked by different offload economics; this is deliberate, not an oversight:

- **Dense**: CPU offload of target layers is expensive (measured: full-GPU ~45-48 t/s vs. ~3-5 t/s
  under heavy offload on the same card). For `priority` in `{balanced, speed}`, the real fraction
  of layers on CPU (`_dense_offload_severity`, 0 = full-GPU) is the **lexicographically dominant**
  sort criterion — a lower-offload option wins regardless of which family (`PRESERVE_CONTEXT_KV_*`
  vs `PRESERVE_KV_PRECISION_CTX_*` vs `TRADEOFF_*`) produced it. At equal severity, larger context
  wins; the family's base rank is only the final tie-break.
- **MoE**: CPU expert offload (`ncmoe`) is cheap (per-token routing; not every expert runs every
  token). For `priority=balanced` it gets a rank **bonus**, not a penalty — MoE offload is a
  first-class option from the start.
- `priority=context` and `priority=quality` (CLI-only; the GUI does not expose them) intentionally
  keep the opposite semantics: they prefer CPU offload over losing context or KV precision.

## Search semantics

### Noise-aware optimization semantics

The optimizer must not rank small benchmark fluctuations as real improvements. Default minimum materiality bands are 5% for confirmed same-context decode, 10% for weak/cross-context decode promotion, and 10%/120 t/s for prefill. Robust FINAL samples and repeated exact-command strong measurements may widen these bands for the current session, but must never narrow the configured values. Search/ranking must preserve decode and context-ingestion as separate axes, use workload elapsed time for real trade-offs, and use Pareto pruning before expensive validation.

A larger ubatch/draft depth is tested only while the previous step shows material marginal value or a single bounded look-ahead is justified. A speculative candidate does not earn FULL/final validation merely for a 1–2 t/s short-sample increase.

A candidate abort is not a session abort.

Candidate outcomes (coarse; see [Candidate lifecycle](#candidate-lifecycle) for the full set):

- PASS
- PASS_DEGRADED
- EARLY_REJECT
- FAILED
- INVALID_ENVIRONMENT
- FATAL

Only unrecoverable environment/model/backend errors should terminate the session globally.

### Decision axes

Search decisions use three separate measured axes, not one blended score:

1. **generation**: robust TG (mixed-workload median/p10 after validation);
2. **context ingestion**: short PP plus harmonic context-staircase fill throughput;
3. **headroom**: sampled minimum free VRAM.

Confirmed PP/TG values inside configured noise bands are equivalent only at the same context. The
intermediate TG interval triggers bounded confirmation when it can change a winner. For trade-offs,
the selected workload profile converts prefill and generation into estimated elapsed seconds.
Noise-aware Pareto pruning removes candidates that are materially worse without compensating wins.

### Coupled-parameter search

Memory placement, ubatch and speculative decoding are not independent dimensions. The optimizer
uses a coordinate/joint search:

1. establish a conservative placement boundary;
2. increase ubatch;
3. if the larger ubatch crosses VRAM/performance limits, move placement toward CPU and retry the
   **same ubatch**;
4. retain stable full-workload candidates;
5. after MTP is enabled, repeat the ubatch/placement search because draft state changes the memory
   boundary;
6. locally tune speculative parameters around the best joint point;
7. preserve the best MTP and non-MTP finalist when both exist;
8. run heterogeneous decode stability and cached context staircase in the same process;
9. rank final profiles by robust TG rather than one short decode sample.

For MoE, recovery increases `ncmoe`. For Dense, recovery reduces `ngl`. Guarded FULL performs the
cheap probe and, if it passes, continues with the staged workload in the same server process,
avoiding a full model reload between probe and validation.

### Robust decode validation

Final candidates are not ranked from one speculative-decode sample. Stability validation runs
heterogeneous deterministic workloads and a shared-prefix cached context staircase in the same
`llama-server` process. Per-request `draft_n`/`draft_n_accepted` come from HTTP timings; exact
`mean len` comes from the matching live server-log line. The report stores raw per-workload/context
rows plus robust summary metrics (median/p10/p90/min/max), not a single TG headline.

## VRAM confidence classes

The hard floor and the preferred reserve answer different questions. Every measured result is
assigned one operating class:

- `REJECT`: below the hard floor; the launch/workload is not usable.
- `FRAGILE`: above the hard floor but below the minimum recommendation threshold; kept as
  diagnostic evidence only.
- `TIGHT`: a FULL/FINAL-proven result inside the explicit 64 MiB hysteresis band below the
  operating floor. Dense workspace is locked and cannot grow automatically.
- `OPERATIONAL`: above the operating floor but below the preferred comfort reserve.
- `SAFE`: at or above the preferred reserve.

MoE is allowed to recover a `TIGHT` point through `ncmoe` because routed-expert CPU placement is a
comparatively cheap axis; Dense never buys workspace from a `TIGHT` target-layer placement. A
confirmed MoE result within 64 MiB / 5% of the preferred reserve is retained as-is — the safety
pass does not buy another CPU expert layer for a comfort-threshold fluctuation.

## Dense search

### CPU-offload policy (hard rule)

**Dense target-layer CPU offload is only ever offered when the model cannot become full-GPU even
at an absolute-minimum `2048`-token context with the most aggressive available KV precision**
(`DENSE_ABSOLUTE_MIN_CONTEXT` in [tuning/target.py](llama_autotuner/tuning/target.py)). This is
checked once per plan (`dense_can_fit_full_gpu_at_minimum`) and gates every `dense-cpu-offload*`
strategy in the solution envelope (`EXACT_TARGET`, `PRESERVE_CONTEXT_KV_*`,
`PRESERVE_KV_PRECISION_CTX_*`, `TRADEOFF_CTX_*`) — not a per-option decision. If the model *can* be
full-GPU at some smaller context/KV, every option that doesn't fit full-GPU at its own context/KV
stays `full-gpu-*` and `INFEASIBLE` rather than being converted to a numeric `ngl`; the context/KV
reduction ladder is extended down to `2048` (Dense only, via `_context_alternatives`'
`extra_floor`) so there is always a real, triable full-GPU point to fall back to. This applies
uniformly across `priority` values, including `context` (which no longer buys the exact requested
context back via offload).

When the gate does trigger (model genuinely cannot fit full-GPU anywhere), the plan's `summary`
carries an explicit warning surfaced to the user before any offload run starts: the model does not
fit at all, offload speed will depend on system RAM bandwidth, confirm this is wanted. Rationale:
measured Dense CPU-offload cost is severe (full-GPU ~45 t/s vs. heavy offload ~3-5 t/s on the same
16 GiB card) — it must never be a silent trade against a comfortable full-GPU alternative that a
smaller context/lower KV precision would have kept available.

### MAX_CONTEXT upsize discovery

Every other Dense envelope family (`PRESERVE_CONTEXT_KV_*`, `PRESERVE_KV_PRECISION_CTX_*`,
`TRADEOFF_CTX_*`) only shrinks context or degrades KV relative to the requested target — none of
them ever propose *more* context than the user asked for. When the exact requested target already
fits full-GPU with real headroom, that left `MAX_CONTEXT` unable to show anything but a duplicate
of `OPTIMAL`, even with several GiB of plainly unused VRAM.

`build_feasibility_plan` generates a bounded **two-tier** doubling ladder of `MAX_CONTEXT_UPSIZE_*`
options up to the model's native context, when `EXACT_TARGET` is `full-gpu` and its measured
headroom is `SAFE` (clears the *preferred* VRAM reserve — the same threshold vram.py's runtime
`VramThresholds` uses, not the stricter static `COMFORTABLE` resource class, which requires ~25% of
total VRAM free: a live run measured 2982 MiB free — well above a 1024 MiB reserve, full requested
context, full KV precision — yet only qualified as `CONSTRAINED`, not `COMFORTABLE`, on a 16 GiB
card):

1. **Q8 tier** (near-zero-risk automatic KV precision) grows from the requested context, doubling
   toward native, until a static estimate says the *hard* VRAM floor would be crossed. This bound
   is deliberately only the hard floor, not the preferred reserve: an earlier version also stopped
   ladder generation when the static estimate's `predicted_dense_ngl != "all"` (a *stricter*,
   reserve-relative signal used elsewhere for the offload-vs-full-GPU decision) — that silently cut
   the ladder short well before the real FRAGILE zone the bisection refine below exists to explore,
   even though `predicted_all_free_mb` (what actually matters for an `ngl=all` upsize candidate)
   was still comfortably above the hard floor.
2. **Q4 tier** only continues from wherever Q8 stopped (never offered as the first-choice tier,
   since Q4 is a materially more aggressive quality trade-off per the KV precision policy above) —
   so context can keep growing toward native even after Q8's VRAM ceiling.

These candidates carry a very low `recommended_rank` so they can never win `OPTIMAL`/`EXACT_TARGET`
ranking; they exist purely for discovery. `AutotuneEngine._discover_max_context_upsize`
quick-probes the combined ladder in ascending context order as the last step of `tune()`, once the
primary target is already confirmed and validated, stopping the whole discovery pass at the first
probe that fails outright or falls below the *hard* VRAM floor. A rung that is runnable but lands
below the *preferred* reserve (`FRAGILE`, not genuinely `SAFE`) is neither accepted as MAX_CONTEXT
nor silently discarded: up to 3 bounded bisection rounds (`_runtime_bracket_context`, the same
interpolation already proven for live FINAL context repair) narrow toward the real knee between it
and the last `SAFE` point in the *same* KV tier — each miss replaces the fragile end of the bracket
and re-interpolates, since a single interpolation across a wide span (e.g. 131072→262144) can itself
land short of the reserve if the real VRAM-vs-context curve isn't perfectly linear across that whole
span. The interpolation target aims `guard_mb=200` past the reserve (not the 16 MiB used elsewhere
for repairing one already-known-fragile point) precisely so it usually converges in one round rather
than three, trading a modest amount of context for reliability. Once a tier's `FRAGILE` miss is
handled (refined to `SAFE`, or the bounded rounds are exhausted), discovery moves on to the next
envelope option (e.g. the next KV tier) rather than stopping the whole pass. It only ever appends
measurements to `self.results` — it never touches `current`/`selected_option`/`target_status`, so it
cannot change what `OPTIMAL` is. `report/generate.py`'s existing "largest successfully measured
context" logic then naturally picks up the largest surviving upsize candidate for `MAX_CONTEXT`.

**Winning context gets one FINAL long-context validation, exactly like every other profile.**
Every measurement above is only a short `recon` probe (~1-2K tokens) — the same weak evidence class
that Phase 6's `FINAL_VALIDATION` exists specifically not to trust for the *requested* target,
because a filled context can still cross the VRAM floor a short probe never reached. This is not
hypothetical: a live GUI session (2026-08-30, Dense+Vision `Qwen3.8-27B-UD-IQ4_XS`) showed a
`MAX_CONTEXT` profile at `ctx=32768` reporting `598 MiB/OPERATIONAL` from a short discovery-style
probe, while the *same exact launch command*'s real `FINAL_VALIDATION` (context filled to ~16K
tokens, run earlier in the same session for the primary target) had already measured `FRAGILE` at
473 MiB. Discovery must not leave itself exempt from the evidence-monotonicity rule that governs
every other profile. After the per-KV-tier loop finds each tier's largest safe rung, the single
largest context across all tiers is run once more with `long_validate=True`
(`phase=MAX_CONTEXT_FINAL_VALIDATION`), the same flag Phase 6 uses. If it holds (`SAFE`/`OPERATIONAL`
and `long_context_passed`), that becomes the FINAL-tier evidence for the same launch key — no report
change needed, since `report/generate.py`'s evidence-rank table already prefers `validation` over
`recon` for an identical command. If it does not hold, the existing full-GPU context-repair
mechanism (`_repair_fragile_dense_full_gpu_context`, already used in Phase 1 for the same "FULL
misses the operational floor" shape) tries one step back toward the last known-safe point in
`self.results`, so `MAX_CONTEXT` ends up reporting a genuinely `FULL`-confirmed context instead of
an unvalidated recon guess. This deliberately costs exactly one extra long-context run (plus, rarely,
one repair confirmation) — the same one-shot discipline Phase 6 itself uses for the primary target,
not a more thorough bar than the rest of the report holds itself to.

**`report/generate.py`'s MAX_CONTEXT selection also had to be fixed for the FINAL-validation step
above to have any user-visible effect.** It previously picked the single largest raw `ctx` among any
non-`REJECT`/`FRAGILE` measurement, with no regard for evidence strength — so a `recon`-tier point
the optimizer's own bisection had explicitly rejected as too fragile could still outrank the smaller,
`validation`-tier point it was refined to and FINAL-validated. `build_profiles` now groups measured
results by `(kv_k, kv_v, ngl, ncmoe, vision, mmproj)` family; within a family, `full`/`validation`
evidence always outranks `recon`/`quick` evidence regardless of raw context size, falling back to
raw-context comparison only when no family member has stronger evidence (this preserves the existing,
intentional LOW-confidence-scout behavior when NORMAL genuinely never FINAL-validated anything larger).

Live-verified end to end, including two real regressions found and fixed while verifying this
feature itself:
- Initial live run: the (now-removed) `predicted_dense_ngl` guard above silently skipped
  `Q8@262144` at plan time, so `MAX_CONTEXT` fell back to the Q4 tier and landed on the model's
  native context anyway (`ctx=262144, KV=q4_0/q4_0, TG=75.9 t/s, SAFE`) — a good outcome, but for
  the wrong reason, and it meant the bisection refine path had never actually been observed firing.
- After removing that guard, `Q8@262144` was correctly generated and measured `FRAGILE` live
  (319-360 MiB free across repeated runs, below the 1024 MiB reserve). The *first* version of the
  refine (single interpolation, `guard_mb=16`) landed on a still-`FRAGILE` point 36-54 MiB short
  twice in a row before a third, narrower attempt finally cleared the reserve
  (`ctx=220160, 1077 MiB free, SAFE`) — which is what motivated both the iterative (up to 3 rounds)
  bisection and the `guard_mb=200` target. With both fixes, a repeat run converged in exactly one
  round: `ctx=215040, KV=q8_0/q8_0, TG=76.5 t/s, VRAM=1250 MiB/SAFE` (vs. `TG=80.6 t/s` at the
  original `ctx=65536` request — 3.3x the requested context for a ~5% decode-speed cost).

Before this whole feature existed, `OPTIMAL`/`MAX_KV_PRECISION`/`FASTEST`/`MAX_CONTEXT` were all
identical on this same model/scenario (`ctx=65536, f16/f16`) — `MAX_CONTEXT` simply had nothing
larger it had ever tried. `OPTIMAL` itself never changes across any of the above; only what
`MAX_CONTEXT` can report does.

1. classify whether the model can become full-GPU after context/KV reduction;
2. when full-GPU is reachable, use the stable full-GPU context/KV Pareto-knee search;
3. when model weights are intrinsically oversized, **do not stop at the first runnable
   requested-context numeric `ngl`**; SCREEN requested and reduced contexts first, because KV
   savings can buy back GPU-resident target layers and materially improve decode;
4. for oversized Dense NORMAL, rank measured `context × KV × ngl` points before any batch sweep and
   FULL-confirm only the selected placement branch;
5. after numeric placement is selected, keep `ngl` fixed while screening batch/ubatch. Do not
   reduce `ngl` merely to increase ubatch unless DEEP exploration proves a material end-to-end
   workload gain;
6. treat target-layer placement primarily as the decode/TG axis and batch/ubatch primarily as the
   prompt-processing/PP axis, while final decisions use representative prefill+generation latency
   rather than either metric alone;
7. if a numeric placement FULL shows a prefill-performance cliff, reject that branch and try the
   next measured context/KV frontier point instead of moving more layers to CPU;
8. if mostly-CPU inference is technically possible but below practical TG, preserve it only as a
   diagnostic/MAX_CONTEXT scout rather than spending the full NORMAL search budget.

Dense solution discovery is hierarchical, not Cartesian, and distinguishes two state machines:

```text
CAN_BECOME_FULL_GPU
  -> stable full-GPU context/KV knees
  -> context-aware REFINE
  -> batch/ubatch
  -> FINAL

OVERSIZED_DENSE (weights cannot fit even at tiny context)
  -> SCREEN requested/reduced context x preferred/Q4 KV
  -> calibrated numeric-ngl placement per point
  -> prune slow/dominated points
  -> choose OPTIMAL / MAX_KV_PRECISION / FASTEST / MAX_CONTEXT frontier roles
  -> FULL-confirm one context/KV/ngl winner
  -> keep ngl fixed while screening ubatch
  -> FINAL with slow-decode token budget when needed
```

For oversized Dense, `context` and `ngl` are coupled: reducing context reduces KV memory and may
restore one or more target layers to GPU, so batch size is a second-order optimization and is never
allowed in NORMAL to silently spend a GPU layer for more PP.

Dense solution discovery also distinguishes `TECHNICAL_CEILING` (can this momentarily fit above the
absolute floor?) from `STABLE_KNEE` (includes an operational uncertainty reserve; the only class
eligible for NORMAL BALANCED/QUALITY deep tuning). The tuning funnel is
`SCREEN -> REFINE -> CONFIRM`: short scouts map feasibility/performance, at most two stable Pareto
knees receive context-aware refinement, and only one NORMAL branch receives expensive
tuning/final confirmation. A runtime-fragile stable-knee attempt is itself sufficient measured
ceiling evidence — NORMAL no longer pays for a separate low-context Q8 anchor or per-KV
technical-ceiling launch.

## MoE search

1. prefer normal layers on GPU;
2. start with conservative expert CPU placement using `-ncmoe`;
3. approach the GPU boundary adaptively;
4. detect the memory-pressure cliff from staged real-server PP;
5. refine around the first safe side of that cliff;
6. only reduce normal GPU layers when expert offload is insufficient for oversized MoE models.

`-ncmoe N` means routed experts from the first N target-model MoE layers remain on CPU, so
`ncmoe=0` is the real full-expert-GPU state and larger values buy VRAM headroom at a possible PP/TG
cost. Full-expert free VRAM and the preferred-reserve seed are stored separately; the optimizer maps
the boundary with QUICK runs and FULL-validates only a finalist. Auxiliary MTP/NextN blocks are not
counted as ordinary target `ncmoe` layers when MTP is disabled.

NORMAL mode treats MoE search as a funnel rather than a Cartesian sweep:

1. **Solution SCREEN**: at most three semantic branches (exact preferred KV, one same-context
   KV-saving branch, one reduced-context preferred-KV branch).
2. **Placement evidence reuse**: a Phase-0 `ncmoe=N PASS / N-1 FAIL` bracket is reused and its old
   512/256 FULL confirmation is deferred.
3. **Joint SCREEN**: ubatch endpoints and expert residency are QUICK-probed. A point below the MoE
   operating floor gets at most a small calibrated `ncmoe` recovery.
4. **Adjacent REFINE**: one safer expert-placement neighbor is QUICK-probed because VRAM pressure
   can make it faster despite more CPU residency.
5. **CONFIRM**: one measured joint winner is normally FULL-confirmed. If two same-context SCREEN
   points differ by 5-10% and headroom would otherwise break the tie, both receive one FULL so the
   ambiguity is resolved rather than labelled noise.
6. **Speculative gate**: MoE MTP starts at n-max=4. No material gain stops immediately. A
   decode-only gain gets one direct FULL and may become FASTEST, while only a workload-latency gain
   (or explicit/DEEP policy) unlocks broader ubatch/n-max/p-min work.

DEEP mode retains broader ubatch/thread/n-max exploration for users who explicitly want exhaustive
characterization.

## Context / KV degradation semantics

There is no implicit context degradation ladder. Lower contexts are separate alternatives and must retain a `capability` degradation label. Lower-precision KV alternatives retain the requested context but carry a `quality-risk` label.

For a Dense model, CPU target-layer offload preserves model/KV semantics but is marked as high-cost `performance` degradation. For MoE, routed-expert offload is modeled separately because the throughput cost can be much smaller.

If a lower context is considered, the feasibility planner creates an explicit `CAPABILITY`-degraded
`SolutionOption` such as `128K -> 96K` or `128K -> 64K`. Reports preserve the original target
status, so a 64K alternative can never be misreported as satisfying a 128K request.

## Vision boundary

Capability/projector presence is checked before launch. Required Vision loads `--mmproj`, and
static planning charges the projector file footprint. Text PP/TG/context remains the optimization
objective. Every Vision candidate loads the real projector, so compatibility and persistent VRAM
cost are authoritative during search — this is a hard static/runtime gate, separate from the FINAL
recognition diagnostic.

Only a FINAL Vision candidate additionally receives one deterministic bundled-image recognition
request (base64 transport for portability, expected code `731`): it records capability evidence,
latency and VRAM pressure, but is **non-gating** — a wrong/missing recognition result lowers report
confidence and does not invalidate the text/context measurements or trigger another placement
branch. `mmproj` load success by itself must not satisfy Vision validation, and image-workload VRAM
samples must contribute to candidate peak/min-free metrics.

## Candidate lifecycle

- `PASS`: full candidate produced useful metrics.
- `PASS_DEGRADED`: technically works but decode is below the practical TG floor for the workload
  (chat=2, agent=5, long-context=5 t/s). The optimizer moves to the next envelope option instead of
  spending budget tuning a confirmed-degraded branch.
- `EARLY_REJECT_MEMORY_CLIFF`: staged 2K/6K/10K prefill collapsed below the configured ratio
  (default 65% of the early PP reference).
- `EARLY_REJECT_ABSOLUTE_VRAM_FLOOR`: sampled free VRAM dropped below the configurable hard floor
  (default 300 MiB). This floor answers only "can this controlled launch continue?" — it is not a
  recommendation threshold; the VRAM confidence classes above gate profiles.
- `EARLY_REJECT_SEVERE_PERFORMANCE_CLIFF`: quick PP/TG collapsed relative to the nearest known-good
  candidate; FULL validation is skipped.
- `FAIL_STARTUP_STALL` / `FAIL_STARTUP_TIMEOUT`: the server never became healthy; this is not a
  placement-memory boundary. A slowly progressing model is not a hang — startup has separate hard
  and progress/stall timeouts.
- `MODEL_STARTUP_FAILED`: repeated split-GGUF bootstrap failure; stop semantic fallbacks and
  preserve an unresolved diagnostic report. Split GGUF is the one exception at the bootstrap
  boundary: all shards are aggregated before planning. No PP/TG measurement exists in this case, so
  context/KV fallbacks must not be labelled infeasible or replayed.
- `FAILED`: startup/OOM/unsupported argument/etc.; optimizer continues.
- `INVALID_ENVIRONMENT`: GPU did not return to baseline or external load changed before launch.

The optimizer never treats a candidate abort as a global autotune abort — candidate, phase and
session termination are separate decisions. Candidate-local FINAL fallback is bounded (NORMAL: at
most two alternatives after the first) and never triggers on `INVALID_ENVIRONMENT`,
environment/startup or exhausted-budget failures.

### Candidate-process ownership

Each candidate server is launched in its own process group and registered with a lease containing
both child and owner PID creation times. Normal CLI interruption unwinds through the server
runner's stop path. The GUI captures the tuner process tree before sending a break signal, waits
for cooperative report/server cleanup, then uses the captured creation-time identities as a bounded
hard-stop fallback. Startup removes only stale owned leases before GPU preflight; the legacy-orphan
matcher is restricted to an orphaned autotuner-shaped command for the selected executable, so an
unrelated GPU workload is never treated as permission to kill arbitrary processes.

### Evidence monotonicity

A stronger workload may invalidate weaker evidence for the same exact launch command. In
particular, a filled-context scout that crosses the VRAM floor or a context-scaling guard removes
the corresponding short SCOUT from stable selection and stable-frontier labels — it may remain
diagnostic evidence, but fallback ordering cannot revive it. Large occupied-cache probes grow a
shared-prefix cache through checkpoints so progress and early catastrophic scaling failure are
observable without duplicating the entire prompt-processing cost.

## Performance parameters in v0.1

Actively searched or varied:

- `-ngl`
- `-ncmoe`
- `-b`
- `-ub`
- `-t`
- `-tb`
- `-c`
- `-ctk/-ctv` baseline (Q8/Q8 in current MVP)
- MTP on/off
- `--spec-draft-n-max`
- `--spec-draft-p-min`

Generated launch commands also include explicit single-GPU CUDA placement, flash attention, load mode, prompt cache, Jinja and metrics when supported.

Sampling/reasoning parameters are intentionally not optimized for quality in v0.1; quality tuning is a separate problem from launch-performance tuning.

## Real-server benchmark workloads

- warmup request (excluded from scoring);
- staged cold PP at approximately 2K / 6K / 10K — if throughput collapses early (e.g. 219 -> 140
  t/s), the candidate is rejected before spending time on the 10K + generation workload;
- coding-oriented TG workload;
- repeat decode measurement and median/variance;
- MTP accepted/generated draft counts when available;
- final heterogeneous decode stability suite (reasoning/code/tool-like/structured/mixed in NORMAL);
- robust TG median/p10/p90/min/max;
- exact MTP mean draft length from the matching live server-log timing line;
- TG<->mean-draft-length and TG<->acceptance correlations;
- shared-prefix cached context staircase. For ctx=65536 NORMAL/DEEP targets ~8K/16K/32K/48K and measures PP+TG without server restart.

The last staircase stage doubles as the final long-context validation, avoiding a duplicate standalone 32K prefill.

## Early termination criteria

### Memory pressure cliff

If staged PP falls below 65% of the early PP reference, reject the candidate without completing expensive decode work.

### Dangerous VRAM

Candidates crossing the emergency free-VRAM floor are rejected even if the server remains alive.

### Ubatch branch stop

When moving beyond the current best causes a large real-server score collapse, stop searching farther in that direction.

### Stall vs slow

A slowly progressing model is not a hang. Startup has separate hard and progress/stall timeouts.

### Run/time budget

Quick/Normal/Deep modes impose maximum server starts and wall-clock budgets. Budget exhaustion must
still produce reports from completed candidates. **These modes are internal search-depth settings
consumed by the CLI/scheduler — see [GUI one-click workflow](#gui-one-click-workflow) for why the
GUI does not expose them as a user-facing choice.**

## Profiles

- OPTIMAL: noise-aware representative workload latency, respecting priority and the normal safety floor.
- MAX_KV_PRECISION: highest measured attention/KV-cache numerical precision.
- FASTEST: highest materially distinct robust decode throughput (65% median + 35% p10), not the highest single lucky TG sample. It may share OPTIMAL's command and must never use the absolute floor as a substitute for recommendation safety.
- MAX_CONTEXT: highest successfully measured non-fragile context; LOW-confidence scout-only evidence is experimental until FINAL validation. For a Dense model whose exact requested target already clears full-GPU at or above the preferred VRAM reserve, this is not necessarily the same context as OPTIMAL: a dedicated discovery phase (`AutotuneEngine._discover_max_context_upsize`) quick-probes a bounded doubling ladder of *larger*-than-requested contexts at a safe Q8 KV tier (never touching OPTIMAL/EXACT_TARGET's exact-quality pick) so MAX_CONTEXT can report a real ceiling instead of duplicating OPTIMAL. See [Dense search § MAX_CONTEXT upsize discovery](#max_context-upsize-discovery).
- FALLBACK: technically runnable but below normal interactive expectations.

Every profile includes a complete launch command and measured evidence. If the same candidate wins
more than one role, it is reported once and labelled with every role it holds (e.g. "OPTIMAL; also
FASTEST") rather than duplicated.

## Persistence and reproducibility

SQLite records sessions and candidate results. Reports include llama.cpp version/build text, hardware, model metadata, baseline VRAM and all accepted/rejected candidates.

A result belongs to a benchmark environment; revalidation keys on model hash, llama.cpp build, GPU/driver and major runtime settings.

A separate shared cross-run history store (`%APPDATA%/LlamaAutotuner/history.sqlite3`) keys on a
model+hardware fingerprint (architecture, size, quantization, expert/block counts; GPU, VRAM,
driver, CPU, **and the llama.cpp build/commit** — `hardware_fingerprint(hardware, llama_version)`
extracts a normalized `build N`/`commit H` signature from the running server's own `--version`
output, so a build upgrade with materially different measured behavior — real upstream examples
have swung PP by 2-3x on Blackwell between commits — gets a fresh key instead of silently reusing
history/calibration captured on a different build; `--use-cached-config`/GUI preview and the actual
search always compute this key the same way, from the same `caps.version_text`/live probe). It has
two tables: `best_configs` persists only the winning candidate per
`(model, hardware, profile)` after a *completed* session, and `observations` persists one row per
*every* distinct candidate config actually probed (including `EARLY_REJECT`/`FAILED`/degraded
points, and interrupted sessions), keyed by the candidate's exact launch parameters, so the store
keeps growing even when a session never reaches a winner. `--use-cached-config` re-validates the
cached OPTIMAL candidate with a single confirmation run instead of a full search, and transparently
falls back to a full search if it no longer validates. `--no-history` opts out of both tables. The
`observations` table is currently write-only from the search's point of view: nothing in the
optimizer reads it back yet to warm-start a new session. See [Roadmap](#roadmap) Phase 1.

A third, narrower table, `hardware_calibration`, *is* read back. It stores the session's
placement-family-level static-VRAM-prediction corrections (`dense:all`, `dense:numeric_median`,
`moe` — never an exact `dense:ngl:N` key, which is model-specific and would not generalize) keyed
by `hardware_key` only, not by model. `AutotuneEngine` already self-calibrates these corrections
*within* one session (`_static_free_corrections_mb`/`_calibrate_from_result`, first probe of a
placement family corrects every later prediction in that run); the CLI now also persists them
after each session and loads any prior value for the current GPU before `tune()` starts
(`prior_calibration_mb`). A real in-session measurement always overrides the loaded prior the
moment it lands (`_predicted_free_for`); the raw uncorrected static formula remains the fallback of
the fallback for hardware with no recorded session yet. This is deliberately scoped smaller than
full `observations` warm-start: it improves the *seed accuracy* for repair/jump decisions
(`_predicted_free_for` call sites) on a brand-new model's first probe when the GPU has been used
before, but does not change `dense_seed_order`/`moe_seed_order` (the very first placement seed,
computed once from the raw `estimate_static_memory` before any correction exists) and does not skip
any live confirmation probe.

## GUI one-click workflow

- The first screen MUST expose only the main target choices: llama.cpp folder/executable, model
  selection, auto-detected Vision/mmproj state, requested context and the `Tune` action.
- The GUI always runs **one fixed search strategy** (`mode=quick`, `priority=balanced`,
  `max_time=8 min`, `max_runs=12`) and surfaces **every** computed Pareto profile
  (OPTIMAL/MAX_CONTEXT/MAX_KV_PRECISION/FASTEST) instead of only one. Search depth and optimization
  priority are intentionally **not** user-facing: the optimizer already computes several trade-off
  profiles per run, so profile diversity comes from that, not from asking the user to pick a search
  mode up front.
- **Hard rule: do not reintroduce a user-facing Quick/Balanced/Thorough or "goal" selector in the
  GUI.** An earlier iteration exposed `Recommended fast` / `Max context` / `Best quality` /
  `Full validation` as a pre-search goal choice; it was deliberately removed because letting the
  user pick a goal before search started routinely picked the wrong branch under VRAM pressure
  (see CHANGELOG "Fix Dense+Vision heavy CPU-offload trap; simplify GUI to one search profile"). If
  the run/time budget itself needs to improve (e.g. `max_runs=12` cutting off MoE+MTP searches
  before `max_time` is spent), fix it inside the scheduler — see Roadmap Phase 0 — never by adding
  a new UI control that fragments search modes again.
- When a report is available, the GUI MUST show the selected recommended command, context, KV,
  `ngl`/`ncmoe`, expected speed, VRAM safety and confidence before the raw log.
- Partial-budget reports MAY be shown as `Fast` confidence; completed strong evidence SHOULD be
  shown as `Validated`.
- The GUI MUST provide a `Copy command` action that copies only the recommended `llama-server`
  launch command.
- The desktop GUI is a thin orchestration layer over the same CLI/autotune engine: it performs
  library discovery and static capability inspection in the UI process, then launches the CLI
  engine as a child process and streams stdout into the GUI log. This keeps GUI and CLI tuning
  semantics identical. GUI ETA is presentation-only and never influences search decisions.

## Known limitations

- Context×KV selection still comes from a precomputed static grid, not a full adaptive bottom-up
  search with intermediate measurements; close to optimal for tested cases but not proven in
  general (a combination of VRAM/model could exist where the grid misses a good near-full-GPU point
  across the tested context range).
- The cross-run history store now records every probed candidate in `observations`, but nothing
  reads that table back yet: warm-starting a new session from prior probes is not wired up, so
  today it is only useful for future analysis/export. Tracked as Roadmap Phase 1 (write path done).
- **Oversized-MoE placement seed leaves VRAM headroom unused and search cost balloons** (found
  2026-08-30, live scenario: Qwen3.8-Flash-Next-UD-Q2_K_XL, 73.45 GiB weights split across 3 shards,
  RTX 5070 Ti 16 GiB, unsloth `b10639-mix-f6f92fe` build). The MoE recon seed started at
  `ncmoe=41-46` (nearly all 48 expert blocks offloaded to CPU), using only ~8-9 GiB of the 16 GiB
  card — several GiB of clearly available VRAM went unused while the offloaded experts (tens of GiB)
  had to live in system RAM instead. Each candidate reload therefore moved tens of GiB through
  RAM/disk, taking 60-180+ seconds just to reach `/health`; the very first probe timed out entirely
  (`WinError 10054`, connection reset, after 180s of cold-disk loading) before a second attempt on
  the same `ncmoe` succeeded once the OS file cache was warm. Stopped by the user after ~10 minutes
  and 4 completed runs rather than let a `--max-time 25` session actually run to completion — at
  this per-candidate cost, a normal search budget is impractical for a model this size on this
  hardware. Not yet fixed — candidate direction: the initial MoE placement seed should more
  aggressively target the *measured* available VRAM (comparable to how the Dense CPU-offload gate
  now checks an absolute-minimum point) rather than starting from a conservative `ncmoe` and walking
  up; per-candidate reload cost should also factor into how many candidates a budget can actually
  afford for very large models.
- Live verification has so far concentrated on one consumer CUDA platform (RTX 5070 Ti, 16 GiB);
  generality across other backends/hardware is unverified.
- **`--vision required` auto-detects a companion `mmproj` only in the exact same directory as the
  selected GGUF; a different quant of the same model in a sibling directory without its own copy
  gets `TARGET_BLOCKED`/`MISSING_REQUIRED_COMPONENT` instead of a Vision-capable run** (found
  2026-08-30: `Qwen3.8-27B-GGUF/` ships `mmproj-BF16.gguf` next to its Dense quants, but the
  separately-downloaded `Qwen3.8-27B-Q4_K_XL-GGUF/` directory does not, even though it is the same
  base model). This is the auto-detect scope working as designed, not a bug — a required capability
  correctly refuses a silent text-only fallback — but a user moving/re-organizing quant directories
  needs to know the projector must travel with the quant, or pass `--mmproj` explicitly.
- ~~QUICK-mode Dense stable-knee search can end in `NO_RECOMMENDABLE_FINAL_CANDIDATE`...~~
  **Resolved 2026-08-30** by the Dense CPU-offload policy change (see
  [Dense search § CPU-offload policy](#cpu-offload-policy-hard-rule)). Found 2026-08-29 on the
  same live scenario (27B Dense + Vision + 262144 requested ctx, RTX 5070 Ti 16 GiB): the QUICK
  full-GPU stable-knee search would commit to a fragile knee (18432 ctx, FRAGILE VRAM class) and,
  on FINAL failure, fall back only within the same full-GPU family instead of reaching the
  `dense-cpu-offload-*` options later in the envelope, ending in an honest but unhelpful "no
  recommendation". Root cause was addressed at the source, not patched around: since Dense CPU
  offload is no longer generated as an option at all (except when the model truly cannot fit
  full-GPU anywhere), there is nothing fragile-and-unreachable left to fall back into — the search
  now simply keeps shrinking context within the full-GPU family. Re-verified live on the same
  scenario: was `PARTIAL` at `ctx≈18432, TG≈14.6-22 t/s, VRAM FRAGILE`; now `COMPLETED` at
  `ctx=4096, ngl=all, KV=f16/f16, TG=41.5 t/s, VRAM OPERATIONAL`, with Vision confirmed PASS.
- ~~QUICK-mode candidate-local FINAL fallback is bounded to one next-frontier attempt and can pick
  a worse recovery than "same KV, smaller context"~~ **Fixed 2026-08-30.** Found on two independent
  live scenarios: (1) Dense+Vision, `--mode quick`, a FULL-confirmed `ctx=16384, KV=q4_0/q4_0`
  failed FINAL and the one allowed fallback jumped to an unrelated `ctx=4096, KV=f16/f16` recon
  candidate instead of retrying the same Q4/Q4 branch smaller, even though `--mode normal` on the
  same scenario found `ctx=16384, KV=q4_0/q4_0, TG=46.4 t/s, COMPLETED` was available all along; (2)
  oversized-Dense (Qwen3.8-27B-UD-Q4_K_XL, 16.35 GiB weights vs. 16 GiB card), where recon had
  already measured `ngl=58` with materially better headroom than `ngl=59` (698 vs 605 MiB free, KV
  precision barely mattering: TG 9.2-9.5 t/s across f16/q8/q4 at the same placement) — FINAL still
  ran at the riskier `ngl=59`, failed at 253 MiB, and the fallback jumped to an unrelated
  `ctx=32768, ngl=57` instead of the already-safer `ngl=58` at the same context, ending `PARTIAL`.
  Root cause: the existing context-repair mechanism (`_repair_fragile_dense_full_gpu_context`,
  [optimizer.py](llama_autotuner/tuning/optimizer.py:1242)) only triggers for a full-GPU (`ngl="all"`),
  `ctx>16384`, *softly* FRAGILE miss — neither scenario qualified (one was `ctx<=16384`, the other
  had `ngl` already numeric and a *hard* `EARLY_REJECT`, not FRAGILE). Fixed by adding a second,
  parallel repair axis in `_run_final_fallbacks` ([optimizer.py:2175](llama_autotuner/tuning/optimizer.py:2175)):
  on any Dense FINAL failure with a recoverable-boundary reason
  (`EARLY_REJECT_ABSOLUTE_VRAM_FLOOR`/`DANGEROUS_VRAM`/`FAIL_OOM`), try one step more conservative
  placement at the *same* context/KV first (reusing the already-tested `_safer_variants`/
  `_guarded_full` machinery) before falling through to the generic frontier queue. Covered by
  `tests/test_v070_hardening.py::test_hard_vram_reject_repairs_one_step_safer_placement_before_frontier_jump`.
  Live-verified only indirectly: a repeat run of the same oversized-Dense scenario completed without
  hitting the EARLY_REJECT branch at all (natural VRAM-baseline variance between runs), so the new
  repair path itself has not yet been *observed* firing live — only proven via the mocked regression
  test that exercises the exact failure it targets.

## Roadmap

Phases are executed and committed one at a time; each phase must not change the ranking/search
behavior of a phase before it unless that is explicitly its purpose.

### Phase 0 — time-aware run budget (done)

`max_runs` is now a soft target, not a hard ceiling: once reached, the engine keeps going as long
as the remaining `max_time` budget can plausibly fit another candidate (estimated from the average
measured cost of runs so far), bounded by a 2x run-count safety ceiling so a session cannot run
away. `TIME_BUDGET_REACHED` is checked first and remains a hard stop regardless of run count. Pure
scheduling change in `AutotuneEngine.budget_reason`/`_time_remaining_for_another_run`
([tuning/optimizer.py](llama_autotuner/tuning/optimizer.py)) — does not touch ranking/search logic.

### Phase 1 — persistent observation cache

- **Write path (done)**: an `observations` table was added to the history store (additive
  migration; the existing `best_configs` table and its behavior are untouched by it). The CLI now
  records every probed candidate result (`HistoryStore.record_observations`) after each session,
  including failed/degraded/interrupted ones, keyed by the same model+hardware fingerprint plus the
  candidate's own launch-parameter key.
- **Read / warm-start path (not started)**: nothing yet reads `observations` back to seed a new
  search's initial placement guess. This is intentionally a separate, later step: it changes what
  the optimizer actually tries first, unlike the write path, which is a pure side effect with zero
  risk to existing ranking/search behavior. When implemented, the analytical estimate must remain
  the source of truth for the guess — the cache only shortcuts how fast search reaches it.

### Phase 2 — long-context curve in ranking (done, pre-existing)

Verified already implemented: `context_decode_tps`/`context_fill_pp` in
[tuning/scoring.py](llama_autotuner/tuning/scoring.py) harmonically blend the measured
`context_staircase` into `profile_decode_tps`/`profile_prefill_tps`/`workload_latency_seconds`,
which is what `choose_preferred`/`pareto_frontier` actually rank on -- so long-context degradation
already participates in OPTIMAL/FASTEST/MAX_CONTEXT selection, not only in diagnostics. The report
also surfaces the retention percentage ("Context staircase TG retained") separately. No code
change was needed; this phase closed as a documentation correction.

### Phase 3 — validation-status surfacing (done, pre-existing)

Verified already implemented: `report/generate.py` prints `Session status` (`COMPLETED` /
`PARTIAL` / `INTERRUPTED` / `FAILED`) and `Stop reason` (e.g. `RUN_BUDGET_REACHED`) at the top of
every report, and the GUI shows a distinct `Confidence: Fast` (provisional/LOW) vs `Validated`
field rather than only logging it. No code change was needed; this phase closed as a documentation
correction.

### Phase 4 — time-to-recommendation as an explicit, separately-tracked metric (not started)

Correctness/stability work (Phases 0-3, plus live verification) must come first and stay separate
from speed work, so a search-shortening change is never evaluated only on "got faster" without also
checking it didn't quietly become less correct. Once live verification confirms the search is
stable and honest (in particular the Dense+Vision edge case found on 2026-08-29 — see
[Known limitations](#known-limitations) — is understood/resolved), open a dedicated phase to reduce
median wall-clock time to a first usable recommendation. Candidate directions, none started yet:

- Measure and publish a "time to first recommendable candidate" number per scenario class
  (Dense/MoE x comfortable/constrained x Vision on/off) as a baseline before optimizing anything.
- The GUI's `RECOMMENDED_PROFILE` (`mode=quick`, 8 min / 12 runs) is already the fast path users
  actually get; NORMAL/DEEP's longer budgets (used for this project's own verification runs, e.g.
  15-20 min per scenario) are not what a normal user experiences and must not be conflated with it
  when judging "is this too slow".
- Warm-start from Phase 1's observation cache (see Phase 1 above) is the most promising lever once
  there is data to warm-start from: skipping already-known-bad probes shortens QUICK mode directly.
  A narrower slice of this — persisted static-VRAM-prediction calibration keyed by GPU only, not by
  exact model — has landed (see the `hardware_calibration` table above); it improves prediction
  accuracy for a *new* model's first probe on already-seen hardware, but does not skip any live
  probe by itself. Full observations-based warm-start (skipping a probe outright because the exact
  candidate was already measured for this model+hardware) is still not started.
- Any latency fix must be verified against the same live scenario matrix used for correctness, not
  just unit tests, since search shortcuts are exactly the kind of change likely to reintroduce a
  correctness regression.

### Explicitly deferred

- User-facing Quick/Balanced/Thorough or goal selector in the GUI — rejected; see
  [GUI one-click workflow](#gui-one-click-workflow).
- Core/frontend package split — pure refactor with no functional payoff until a second frontend
  (e.g. an external planner adapter) actually needs it.
- External benchmark prior / bundled knowledge database — needs its own security review (never
  execute scraped text as a shell command) and a curation pipeline; a separate initiative, not
  near-term.
- ML residual performance predictor — meaningless before Phase 1's observation cache has enough
  data.
- Optuna (or llama-optimus, which is a thin wrapper around it) as the search engine — researched
  2026-08-30, rejected: our per-phase trial budgets (2-4 probe points, ~9-17 real candidates per
  whole session) are far below where TPE earns its keep over a well-tuned domain heuristic, real
  trials cost 10-180s each (not the ML-training-hyperparameter-tuning regime Optuna targets), and it
  would add 7 new dependencies (incl. numpy, sqlalchemy) to a currently single-dependency (`psutil`)
  tool. See [RESEARCH_OPTUNA_INTEGRATION.md](RESEARCH_OPTUNA_INTEGRATION.md) for the full analysis.
  One idea from llama-optimus is worth a follow-up on its own: an explicit pre-search warm-up phase
  to avoid trusting a "cold hardware" first measurement — not yet investigated on our hardware.

### Longer-term

- NVIDIA multi-GPU with split mode, tensor split, main GPU, asymmetric VRAM and topology;
- external draft models and additional speculative decoding strategies;
- richer Vision workloads beyond the final bundled-image diagnostic;
- AMD ROCm/Vulkan, Intel, Metal and CPU-only tuning after each backend has reliable telemetry and reference hardware.

### Verification plan — large offloaded-MoE context ladder (planned, not started)

Everything measured on Flash-Next so far (see Known limitations and the 2026-08-30 CHANGELOG
entries) used `ctx=4096`/`8192` specifically to keep the smoke/completion tests tractable given the
~150-180s/candidate reload cost — not because a larger context was tried and failed. For a model
this large, more context is a materially different trade-off than for a comfortably-resident Dense
model: every extra token of KV cache and every extra MiB of compute-buffer competes directly with
the handful of GiB left after tens of GiB of expert weights are already pinned across GPU+CPU, so
`ncmoe` may need to grow (more experts pushed to RAM) purely to make room for a larger context, not
because the search "wants" to be more conservative. This has never actually been measured across a
context range. The goal of this plan is to characterize that curve on one large offloaded-MoE model
before drawing any conclusion about how well the tool handles this class of model in general.

1. **Subject and baseline.** Reuse Qwen3.8-Flash-Next-UD-Q2_K_XL (73.45 GiB, 3-shard split GGUF,
   same RTX 5070 Ti 16 GiB) so results are directly comparable to the existing `ctx=8192` completed
   run (`OPTIMAL: ncmoe=41, TG=20.2, PP=501.4, 4099 MiB/SAFE`, 12 runs/39:43) already on record.
2. **Context ladder, one context per session, run to actual completion (not a bounded smoke test)**:
   `ctx=16384`, `32768`, `65536`. Do not skip straight to 65536 -- the whole point is the curve, not
   just the endpoint, and a monotonic-but-nonlinear relationship between context and `ncmoe`/VRAM is
   the expected (not exceptional) outcome for this model class.
3. **Per-context, record explicitly** (not just OPTIMAL): `ncmoe` chosen, `vram_free_min_mb` and
   operating class, `tg_tps`/`pp_tps`, `ram_peak_mb`, number of runs and total wall-clock time, and
   whether `_predicted_free_for`'s static estimate for the winning candidate was close to the
   measured value (this is also a chance to sanity-check the static MoE KV/weight formulas at a
   scale far outside anything they have been checked against so far).
4. **Budget expectation, stated up front so it is not a surprise mid-run**: if reload cost dominates
   the way it did at `ctx=8192` (roughly 165s/candidate average across 12 runs), a similarly-sized
   completed search at each higher context could take a comparable ~35-45 minutes; the full
   4-point ladder (8192 already done + 16384/32768/65536) is therefore a multi-hour undertaking, not
   a quick follow-up. Confirm with the user before launching the next context step rather than
   chaining all of them unattended.
5. **Secondary, optional axis once the ladder itself is understood**: `--no-mmap` was flagged (from
   an external llama.cpp optimization write-up reviewed 2026-08-30, not yet independently verified
   against our own build) as a possible fix for MoE TG jitter from page faults on chaotic expert-weight
   reads under `mmap`. If the context ladder shows meaningful decode variance for this model, a
   `mmap` vs `--no-mmap` A/B at the ladder's most representative context is a reasonable next step --
   but only after the ladder itself, and only if variance is actually observed, not speculatively.
6. **Explicitly out of scope for this plan**: fixing the "oversized-MoE placement seed leaves VRAM
   headroom unused" limitation itself (a separate, already-documented code-change item above with
   its own candidate direction) -- this plan is about *measuring* the real curve first, which is also
   the evidence that change would need to be judged against.

## v0.5.3 additional acceptance criteria

- GUI MUST display exact elapsed time while tuning.
- GUI SHOULD display a clearly approximate remaining-time estimate; it MUST NOT claim exact ETA for an adaptive branch.
- Finished CLI and reports MUST contain total elapsed autotune time.
- A Vision candidate MUST process at least one actual image request before it can be accepted.
- The bundled recognition fixture MUST require semantic image input (expected code `731`) and its latency/result MUST be recorded.
- Image-workload VRAM samples MUST contribute to candidate peak/min-free metrics.
- `mmproj` load success by itself MUST NOT satisfy Vision validation.

## v0.5.4 core acceptance criteria

1. A Dense option may remain full-GPU whenever predicted/observed free VRAM is above the absolute floor, even if it misses the preferred reserve.
2. Missing the preferred reserve alone must never authorize Dense CPU layer offload.
3. When exact preferred-quality full-GPU is unavailable and multiple full-GPU trade-offs are runnable, the engine must cheaply measure a bounded shortlist before deep tuning.
4. A materially faster measured solution may override static trade-off ordering; performance-equivalent solutions retain the user's degradation priority.
5. Context and KV trade-offs may be combined to preserve Pareto/knee candidates.
6. Vision image-recognition correctness must not invalidate the core runtime search; `mmproj` startup and real VRAM use remain mandatory for Vision-required candidates.
7. Numeric GUI ETA must not be presented during highly adaptive pre-FINAL search.

## v0.5.5 MoE core acceptance criteria

1. `EXACT_TARGET/full-gpu` for MoE is legal only when `ncmoe=0` clears the absolute VRAM floor.
2. If expert offload is required, it is disclosed as a performance trade-off and starts from the minimum predicted `ncmoe` that clears the hard floor, not the preferred reserve.
3. Coarse MoE placement uses QUICK probes until the boundary is mapped; intermediate headroom points do not automatically earn FULL runs.
4. The requested exact target participates in solution reconnaissance whenever statically runnable.
5. A large cold load must not be classified as stalled solely because it exceeds a fixed 30-second no-log window.

## Dense NORMAL search budget policy (v0.5.6)

- Automatic context reduction for large targets must include 32K and 16K preferred-KV candidates when allowed.
- Solution-level reconnaissance uses a micro benchmark and may be reused as the identical placement probe.
- Preferred VRAM reserve is not a hard requirement; the absolute floor is.
- A numeric Dense placement calibration must never be applied to `ngl=all`.
- Batch/ubatch candidates require a material quick improvement before another FULL confirmation.
- NORMAL initially performs one FINAL validation for the selected Dense branch. If that candidate fails for a candidate-local reason, a bounded next-frontier fallback chain may FULL-confirm and FINAL-validate at most two alternatives. Environment/startup/budget failures do not trigger semantic fallback. DEEP may validate more frontier candidates.
- Unselected measured solution scouts remain visible in the report and must be labelled as lower-confidence scouts, not final validated profiles.
