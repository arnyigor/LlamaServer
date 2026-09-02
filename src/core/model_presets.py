"""Model capability presets: thinking format and chat-template kwargs.

Used by ``cli_builder.build_args`` to decide how thinking/reasoning is emitted
for a given model. Most models use the modern ``--reasoning`` / ``--reasoning-effort``
flags; Qwen3.5+ (and similar) require ``--chat-template-kwargs enable_thinking=...``
instead.

Matching is by architecture (from the model scan cache) or by a filename
substring in the model path. ``build_args`` resolves this from ``model_path``
alone; callers may pass a pre-resolved ``model_compat`` for precision.
"""

from typing import Dict, List, Optional

# Each preset: id/name for diagnostics, ``match`` (architecture list and/or
# filename substrings, case-insensitive), and ``compat`` (emission profile).
_MODEL_PRESETS: List[Dict] = [
    {
        "id": "qwen3.5+",
        "name": "Qwen3.5+ (chat-template thinking)",
        "match": {
            "architecture": ["qwen35", "qwen35moe"],
            "name_substring": [
                "qwen3.5",
                "qwen3.6",
                "qwen3.7",
                "qwen3.8",
                "qwen3.9",
            ],
        },
        "compat": {
            "thinking_format": "chat-template",
            "chat_template_kwargs": {
                "enable_thinking": {"var": "thinking.enabled"},
                "reasoning_effort": {"var": "thinking.effort", "omit_when_off": True},
            },
        },
    },
]


def resolve_model_compat(
    model_path: str, architecture: Optional[str] = None
) -> Optional[Dict]:
    """Return the ``compat`` profile for a model, or ``None`` if unknown.

    ``architecture`` (e.g. from the scan cache) takes precedence; otherwise the
    model path/filename is matched against the preset substrings.
    """
    if not model_path and not architecture:
        return None
    path_l = (model_path or "").lower()
    arch_l = (architecture or "").lower()
    for preset in _MODEL_PRESETS:
        m = preset.get("match", {})
        archs = [a.lower() for a in m.get("architecture", [])]
        subs = [s.lower() for s in m.get("name_substring", [])]
        if arch_l and arch_l in archs:
            return preset["compat"]
        if any(s in path_l for s in subs):
            return preset["compat"]
    return None
