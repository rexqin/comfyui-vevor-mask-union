"""Unified {{placeholder}} prompt rendering and catalog (design-agent render_template parity)."""

from __future__ import annotations

import json
import os
import re
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_CATALOG_PATH = os.path.join(_HERE, "generative_prompt_catalog.json")
_LANG_PATCHES_PATH = os.path.join(_HERE, "generative_lang_patches.json")
_RETRY_PROMPTS_PATH = os.path.join(_HERE, "generative_retry_prompts.json")
_IMAGE_TEMPLATE_PATH = os.path.join(_HERE, "generative_prompt_image.txt")

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")

PROMPT_KEYS = ("analyze", "translate", "review", "image")


def render_template(template: str, variables: dict[str, Any] | None) -> str:
    """Replace ``{{name}}`` with string values; unknown keys become empty."""
    text = str(template or "")
    vars_map = variables if isinstance(variables, dict) else {}

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        val = vars_map.get(key)
        if val is None:
            return ""
        return str(val)

    return _PLACEHOLDER_RE.sub(repl, text)


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def load_catalog() -> dict[str, str]:
    with open(_CATALOG_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    out = {str(k): str(v) for k, v in data.items() if k in PROMPT_KEYS and v}
    if "image" not in out and os.path.isfile(_IMAGE_TEMPLATE_PATH):
        out["image"] = _read_text(_IMAGE_TEMPLATE_PATH)
    return out


def resolve_template(prompt_key: str, template_override: str = "") -> str:
    custom = (template_override or "").strip()
    if custom:
        return custom
    key = str(prompt_key or "").strip().lower()
    catalog = load_catalog()
    if key in catalog:
        return catalog[key]
    raise ValueError(f"未知 prompt_key {prompt_key!r}；允许: {', '.join(PROMPT_KEYS)}")


def load_lang_patches() -> dict[str, list[str]]:
    if not os.path.isfile(_LANG_PATCHES_PATH):
        return {}
    with open(_LANG_PATCHES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[str]] = {}
    for code, lines in data.items():
        if isinstance(lines, list):
            out[str(code).lower()] = [str(x) for x in lines if str(x).strip()]
    return out


def load_retry_prompt_parts() -> dict[str, str]:
    if not os.path.isfile(_RETRY_PROMPTS_PATH):
        return {}
    with open(_RETRY_PROMPTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def parse_variables_json(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"variables JSON 无效: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("variables 必须是 JSON 对象")
    return obj


def format_lang_section(target_lang: str, lang_code: str) -> str:
    patches = load_lang_patches()
    code = str(lang_code or "").lower()
    lines = patches.get(code) or []
    if not lines:
        return ""
    body = "\n".join(f"- {ln}" for ln in lines)
    return f"\n\n【{target_lang} 专属规范】\n{body}"


def format_retry_section(part_key: str, body: str) -> str:
    if not str(body or "").strip():
        return ""
    parts = load_retry_prompt_parts()
    tpl = parts.get(part_key, "")
    if not tpl:
        return str(body)
    return render_template(tpl, {"body": body})
