"""Comfy nodes for generative image-text translation (aligned with demo/pipeline.mjs)."""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from generative_prompts import (  # noqa: E402
    build_image_prompt,
    build_paragraph_section,
    build_translate_prompt,
    load_analyze_prompt,
)

# Ported from demo/lib-text.mjs
NL_MARK = "\\n"
PAD_BORROW_RATIO = 0.45
CONTAINER_CAP_RATIO = 0.96
NO_CONTAINER_EXTRA = 0.05
MAX_ALLOW_RATIO = 1.0
LIMIT_LABEL_THRESHOLD = 0.25
LENGTH_SLACK_RATIO = 0.03
LENGTH_SLACK_MIN = 1
LINE_STRICT_MAX_CHARS = 120
REFLOW_MIN_CHARS = 60
GLYPH_CHARS_PER_ROW = 9

LANG_WIDTH_FACTOR: dict[str, float] = {
    "de": 1.08,
    "nl": 1.06,
    "fr": 1.05,
    "es": 1.02,
    "it": 1.02,
    "pt": 1.02,
    "pl": 1.06,
}

ALIGN_TAG = {"left": "left-aligned", "center": "centered", "right": "right-aligned"}

COLOR_NAMES: list[tuple[str, tuple[int, int, int]]] = [
    ("white", (255, 255, 255)),
    ("black", (0, 0, 0)),
    ("dark gray", (60, 60, 60)),
    ("gray", (128, 128, 128)),
    ("light gray", (205, 205, 205)),
    ("orange", (245, 90, 25)),
    ("red", (215, 45, 45)),
    ("yellow", (250, 200, 20)),
    ("green", (45, 160, 75)),
    ("blue", (45, 95, 210)),
    ("brown", (125, 85, 45)),
    ("beige", (235, 220, 190)),
    ("pink", (240, 130, 165)),
    ("purple", (130, 70, 180)),
]

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
def _resolve_font_path(font_dir: str) -> str | None:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if font_dir:
        candidates.append(font_dir)
        candidates.append(os.path.join(here, font_dir))
    candidates.append(os.path.join(here, "fonts"))
    for path in candidates:
        if not path or not os.path.isdir(path):
            continue
        for name in ("InterTight-Black.ttf", "InterTight-Bold.ttf", "InterTight-Regular.ttf"):
            full = os.path.join(path, name)
            if os.path.isfile(full):
                return full
    return None


def _load_font(font_dir: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(1, int(size))
    key = (font_dir or "", size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    path = _resolve_font_path(font_dir)
    font = ImageFont.load_default()
    if path:
        try:
            font = ImageFont.truetype(path, size)
        except OSError:
            pass
    _FONT_CACHE[key] = font
    return font


def measure_em(text: str, font_dir: str) -> float:
    font = _load_font(font_dir, 1000)
    if hasattr(font, "getlength"):
        return float(font.getlength(text)) / 1000.0
    return len(text) * 0.55


def _parse_json_array(text: str) -> list:
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.I).strip()
    raw = re.sub(r"```$", "", raw).strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(raw[start : end + 1])
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def _parse_translation_map(text: str) -> dict[int, str]:
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.I).strip()
    raw = re.sub(r"```$", "", raw).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    out: dict[int, str] = {}
    if start < 0 or end <= start:
        return out
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return out
    if not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        out[idx] = str(v or "").replace("\\n", "\n").strip()
    return out


def _as_box(v) -> list[float] | None:
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    nums = [float(x) for x in v]
    if any(not np.isfinite(n) for n in nums):
        return None
    return nums


def normalize_blocks(llm_text: str) -> list[dict]:
    blocks = []
    for i, row in enumerate(_parse_json_array(llm_text)):
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "")
        bbox = _as_box(row.get("bbox"))
        if not text or not bbox:
            continue
        role = str(row.get("role") or "").lower()
        blocks.append(
            {
                "id": i + 1,
                "text": text,
                "translation": None,
                "role": role if role in ("title", "bullet", "label", "paragraph") else None,
                "bbox": bbox,
                "containerBbox": _as_box(row.get("containerBbox")),
                "align": row.get("align") if row.get("align") in ("left", "center", "right") else "left",
                "bold": bool(row.get("bold")),
                "sizePct": float(row.get("sizePct") or 5),
                "textColor": str(row.get("textColor") or "#FFFFFF"),
                "colorRuns": row.get("colorRuns") if isinstance(row.get("colorRuns"), list) else [],
                "containerColor": str(row.get("containerColor") or ""),
                "isLogo": bool(row.get("isLogo")),
            }
        )
    return blocks


def compute_space_budget(block: dict) -> dict:
    text_w = max(0.005, float(block["bbox"][2]))
    if block.get("containerBbox"):
        cw = max(float(block["containerBbox"][2]), text_w)
        borrow = max(0.0, cw - text_w) * PAD_BORROW_RATIO
        avail_w = min(text_w + borrow, cw * CONTAINER_CAP_RATIO)
    else:
        avail_w = text_w * (1 + NO_CONTAINER_EXTRA)
    allow_ratio = min(MAX_ALLOW_RATIO, max(0.0, avail_w / text_w - 1))
    return {"allowRatio": allow_ratio, "label": allow_ratio < LIMIT_LABEL_THRESHOLD}


def line_capacity(block: dict, allow_ratio: float, lang_factor: float = 1.0) -> dict:
    lines = [ln.strip() for ln in block["text"].split("\n") if ln.strip()]
    max_len = max(1, max((len(ln) for ln in lines), default=1))
    limit = max(1, round(max(max_len, round(max_len * (1 + allow_ratio))) / max(lang_factor, 0.01)))
    slack = max(LENGTH_SLACK_MIN, round(limit * LENGTH_SLACK_RATIO))
    return {"limit": limit, "pass": limit + slack}


def paragraph_capacity(block: dict, allow_ratio: float) -> dict:
    lines = len([ln for ln in block["text"].split("\n") if ln.strip()]) or 1
    per = line_capacity(block, allow_ratio)
    limit = per["limit"] * lines
    slack = max(LENGTH_SLACK_MIN, round(limit * LENGTH_SLACK_RATIO))
    return {"limit": limit, "pass": limit + slack, "lines": lines}


def fill_lines(text: str, capacity: float, font_dir: str) -> list[str]:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return []
    if not (capacity > 0):
        return [clean]
    space = measure_em(" ", font_dir)
    words = clean.split(" ")
    lines: list[str] = []
    current = ""
    width = 0.0
    for word in words:
        w = measure_em(word, font_dir)
        if not current:
            current, width = word, w
            continue
        if width + space + w <= capacity:
            current += f" {word}"
            width += space + w
        else:
            lines.append(current)
            current, width = word, w
    if current:
        lines.append(current)
    return lines


def wrap_to_original_width(text: str, original_text: str, font_dir: str) -> list[str]:
    src = [ln.strip() for ln in str(original_text or "").split("\n") if ln.strip()]
    capacity = max((measure_em(ln, font_dir) for ln in src), default=0.0) if src else 0.0
    return fill_lines(text, capacity, font_dir)


def wrap_target(block: dict, font_dir: str) -> str:
    text = str(block.get("translation") or "")
    if not text.strip():
        return text
    src_lines = [ln for ln in str(block.get("text") or "").split("\n") if ln.strip()]
    flat = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    if len(src_lines) >= 2 and len(flat) >= REFLOW_MIN_CHARS:
        lines = wrap_to_original_width(flat, block["text"], font_dir)
        if lines and len(lines) <= len(src_lines):
            return "\n".join(lines)
    if "\n" in text:
        return text
    lines = wrap_to_original_width(text, block["text"], font_dir)
    return "\n".join(lines) if len(lines) > 1 else text


def blank_unchanged_lines(from_text: str, to_text: str) -> tuple[str, str]:
    def same(a: str, b: str) -> bool:
        return re.sub(r"\s+", " ", a).strip().lower() == re.sub(r"\s+", " ", b).strip().lower()

    src = str(from_text or "").split("\n")
    dst = str(to_text or "").split("\n")
    if len(src) != len(dst) or len(src) < 2:
        return from_text, to_text
    f_out, t_out, changed = [], [], 0
    for s, d in zip(src, dst):
        if same(s, d):
            f_out.append("")
            t_out.append("")
        else:
            f_out.append(s)
            t_out.append(d)
            changed += 1
    if changed == 0:
        return from_text, to_text
    return "\n".join(f_out), "\n".join(t_out)


def is_translatable(block: dict) -> bool:
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", str(s or "")).strip().lower()

    return (
        not block.get("isLogo")
        and bool(str(block.get("translation") or "").strip())
        and norm(block.get("translation")) != norm(block.get("text"))
    )


def _pct(n: float) -> str:
    return f"{n * 100:.1f}%"


def _color_name_of(text_color: str) -> str | None:
    m = re.search(r"#?([0-9a-f]{6})", str(text_color or "").strip(), re.I)
    if not m:
        return None
    v = int(m.group(1), 16)
    r, g, b = (v >> 16) & 255, (v >> 8) & 255, v & 255
    best, best_d = None, float("inf")
    for name, (cr, cg, cb) in COLOR_NAMES:
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < best_d:
            best_d, best = d, name
    return best


def _tag_of(name: str, box: list[float] | None) -> str:
    if not box:
        return ""
    return f"[{name}:{','.join(_pct(v) for v in box)}] "


def _safe_tag_of(block: dict) -> str:
    cb = _as_box(block.get("containerBbox"))
    bb = _as_box(block.get("bbox"))
    if not cb or not bb:
        return ""
    bx, _, bw, _ = bb
    cx, _, cw, _ = cb
    left = max(0.0, bx - cx)
    right = max(0.0, cx + cw - (bx + bw))
    return f"[safe:{_pct(left)}/{_pct(right)}] "


def _group_key_of(block: dict) -> str:
    return "|".join(
        [
            str(block.get("role") or ""),
            _color_name_of(block.get("textColor") or "") or "-",
            str(block.get("containerColor") or "-"),
            str(block.get("align") or "left"),
            str(round(float(block.get("sizePct") or 0))),
        ]
    )


def _lines_of(text: str) -> int:
    return len([ln for ln in text.split("\n") if ln.strip()]) or 1


def build_image_rows(blocks: list[dict], font_dir: str) -> str:
    targets = [b for b in blocks if is_translatable(b)]
    group_sizes: dict[str, int] = {}
    for b in targets:
        key = _group_key_of(b)
        group_sizes[key] = group_sizes.get(key, 0) + 1
    group_nos: dict[str, int] = {}
    next_group_no = 0

    rows = []
    for b in targets:
        align = str(b.get("align") or "left")
        align_tag = f"[{ALIGN_TAG.get(align, align)}] " if align in ALIGN_TAG else f"[align:{align}] "
        name = _color_name_of(b.get("textColor") or "")
        color_tag = f"[{name}] " if name else ""
        bold_tag = "[bold] " if b.get("bold") else ""
        max_lines = b.get("maxLines") or _lines_of(b["text"])
        lines_tag = f"[lines:{max_lines}] "
        box_tag = _tag_of("box", b.get("bbox"))
        panel_tag = _tag_of("panel", b.get("containerBbox"))
        safe_tag = _safe_tag_of(b)
        key = _group_key_of(b)
        group_tag = ""
        if group_sizes.get(key, 0) >= 2:
            if key not in group_nos:
                next_group_no += 1
                group_nos[key] = next_group_no
            group_tag = f"[group:g{group_nos[key]}] "

        wrapped = wrap_target(b, font_dir)
        pair_from, pair_to = blank_unchanged_lines(b["text"], wrapped)
        from_s = pair_from.replace("\n", NL_MARK)
        to_s = pair_to.replace("\n", NL_MARK)
        rows.append(
            f"- {align_tag}{color_tag}{bold_tag}{lines_tag}{box_tag}{panel_tag}{safe_tag}{group_tag}"
            f'"{from_s}" -> "{to_s}"'
        )
    return "\n".join(rows)


def build_glyph_chars(blocks: list[dict]) -> str:
    text = "".join(str(b.get("translation") or "") for b in blocks if is_translatable(b))
    uniq = sorted({c for c in text if c.strip()}, key=lambda c: c)
    accent = re.compile(r"[áéíóúñüÁÉÍÓÚÑÜàâäçèêëîïôöùûÿœæß]")

    def is_accent(c: str) -> bool:
        return bool(accent.search(c))

    groups = [
        "".join(sorted(c for c in uniq if c.isupper() and not is_accent(c))),
        "".join(sorted(c for c in uniq if c.islower() and not is_accent(c))),
        "".join(sorted(c for c in uniq if c.isdigit()))
        + "".join(c for c in uniq if not (c.isalpha() or c.isdigit())),
        "".join(sorted(c for c in uniq if is_accent(c))),
    ]
    groups = [g for g in groups if g]
    rows: list[str] = []
    for g in groups:
        for i in range(0, len(g), GLYPH_CHARS_PER_ROW):
            rows.append(g[i : i + GLYPH_CHARS_PER_ROW])
    return "\n".join(rows)


def render_glyph_card(font_dir: str, chars: str, font_size: int = 220) -> np.ndarray:
    font = _load_font(font_dir, font_size)
    lines = [ln.rstrip() for ln in str(chars or "").split("\n") if ln.strip()]
    if not lines:
        lines = ["C4%"]
    pad = round(font_size * 0.09)
    line_h = round(font_size * 1.25)

    def line_width(line: str) -> float:
        return float(font.getlength(line)) if hasattr(font, "getlength") else len(line) * font_size * 0.55

    w = int(max(line_width(ln) for ln in lines) + pad * 2)
    h = line_h * len(lines) + pad
    img = Image.new("RGB", (max(w, 32), max(h, 32)), color=(0x14, 0x14, 0x14))
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        y = pad + i * line_h
        draw.text((pad, y), line, fill=(255, 255, 255), font=font)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


def _blocks_as_list(blocks) -> list[dict]:
    if blocks is None:
        return []
    if isinstance(blocks, str):
        try:
            blocks = json.loads(blocks)
        except json.JSONDecodeError:
            return []
    return blocks if isinstance(blocks, list) else []


class VevorGenerativeAnalyzePrompt:
    """Full vision-analyze instruction (same as demo ``ANALYZE_PROMPT``)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schema_template": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "留空=包内 generative_analyze_prompt.txt；可覆写版面 JSON 字段说明",
                    },
                ),
                "user_line": (
                    "STRING",
                    {
                        "default": "识别这张图上所有可见文字块，只输出 JSON 数组。",
                        "multiline": False,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build"
    CATEGORY = "vevor/generative"

    def build(self, schema_template: str, user_line: str):
        body = (schema_template or "").strip() or load_analyze_prompt()
        return (f"{body}\n\n{user_line.strip()}",)


class VevorGenerativeParseBlocks:
    """Parse LLM JSON array into normalized layout blocks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"analyze_response": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("JSON",)
    RETURN_NAMES = ("blocks",)
    FUNCTION = "parse"
    CATEGORY = "vevor/generative"

    def parse(self, analyze_response: str):
        return (normalize_blocks(analyze_response),)


class VevorGenerativeTranslateLexicon:
    """Glossary section for translate prompt (design-agent ``glossary.ts``)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blocks": ("JSON", {"forceInput": True}),
                "lang_code": (["de", "fr", "it", "es"],),
                "glossary_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": '可选：[{"from":"…","lang":"de|*","to":"…"}]',
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lexicon_extra",)
    FUNCTION = "build"
    CATEGORY = "vevor/generative"

    def build(self, blocks, lang_code: str, glossary_json: str = ""):
        raw = str(glossary_json or "").strip()
        entries: list[dict] = []
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    entries = [e for e in parsed if isinstance(e, dict)]
            except json.JSONDecodeError:
                entries = []
        items = [
            {"text": str(b.get("text") or "")}
            for b in _blocks_as_list(blocks)
            if not b.get("isLogo") and str(b.get("text") or "").strip()
        ]
        if not items or not entries:
            return ("",)
        code = str(lang_code or "").lower()
        hits = []
        for e in entries:
            lang = str(e.get("lang") or "*").strip().lower()
            if lang != "*" and lang != code:
                continue
            from_text = str(e.get("from") or "").strip()
            to_text = str(e.get("to") or "").strip()
            if not from_text or not to_text:
                continue
            needle = from_text.lower()
            if any(needle in str(it.get("text") or "").lower() for it in items):
                hits.append((from_text, to_text))
        if not hits:
            return ("",)
        body = "\n".join(f'- "{a}" 必须译为 "{b}"' for a, b in hits)
        section = (
            "\n\n【术语表（以下词条必须严格采用指定译法，不得改动）】\n" + body
        )
        return (section,)


class VevorGenerativeTranslatePrompt:
    """Build translation prompt with spatial budget hints (demo ``lib-api``)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blocks": ("JSON", {"forceInput": True}),
                "target_lang": ("STRING", {"default": "德语"}),
                "rules_template": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "留空=内置规则；占位符 {lang} {extra} {rows}",
                    },
                ),
            },
            "optional": {
                "lang_code": (["de", "fr", "it", "es"],),
                "lexicon_extra": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build"
    CATEGORY = "vevor/generative"

    def build(
        self,
        blocks,
        target_lang: str,
        rules_template: str = "",
        lang_code: str = "de",
        lexicon_extra: str = "",
    ):
        lang_factor = LANG_WIDTH_FACTOR.get(str(lang_code or "").lower(), 1.0)
        items = []
        for b in _blocks_as_list(blocks):
            if b.get("isLogo") or not str(b.get("text") or "").strip():
                continue
            budget = compute_space_budget(b)
            text = b["text"]
            if len(text) > LINE_STRICT_MAX_CHARS:
                cap = paragraph_capacity(b, budget["allowRatio"])
                items.append(
                    {
                        "id": b["id"],
                        "text": text,
                        "role": b.get("role"),
                        "wholeCap": cap["limit"],
                        "lines": cap["lines"],
                    }
                )
            else:
                cap = (
                    line_capacity(b, budget["allowRatio"], lang_factor)
                    if budget["label"]
                    else None
                )
                items.append(
                    {
                        "id": b["id"],
                        "text": text,
                        "role": b.get("role"),
                        "lineCap": cap["limit"] if cap else None,
                    }
                )
        rows = []
        for it in items:
            role = f"[{it['role']}] " if it.get("role") else ""
            line_hint = f"（每行不超过 {it['lineCap']} 字符）" if it.get("lineCap") else ""
            whole_hint = (
                f"（整段不超过 {it['wholeCap']} 字符，保持 {it['lines']} 行）" if it.get("wholeCap") else ""
            )
            escaped_text = it["text"].replace("\n", "\\n")
            rows.append(f'{it["id"]}. {role}"{escaped_text}"{line_hint}{whole_hint}')
        extra = build_paragraph_section(items) + str(lexicon_extra or "")
        prompt = build_translate_prompt("\n".join(rows), target_lang, extra, rules_template)
        return (prompt,)


class VevorGenerativeApplyTranslations:
    """Merge translation LLM JSON into blocks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blocks": ("JSON", {"forceInput": True}),
                "translation_response": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("JSON",)
    RETURN_NAMES = ("blocks",)
    FUNCTION = "apply"
    CATEGORY = "vevor/generative"

    def apply(self, blocks, translation_response: str):
        data = [dict(b) for b in _blocks_as_list(blocks)]
        mapping = _parse_translation_map(translation_response)
        for b in data:
            tr = mapping.get(int(b.get("id", 0)))
            if tr:
                b["translation"] = tr
        return (data,)


def _norm_source_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _parse_translation_override_map(raw: str) -> dict[str, str]:
    """API / widget JSON → normalized source → translation."""
    out: dict[str, str] = {}
    text = str(raw or "").strip()
    if not text:
        return out
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return out
    items = obj.get("items") if isinstance(obj, dict) else obj
    if not isinstance(items, list):
        return out
    for row in items:
        if not isinstance(row, dict):
            continue
        source = _norm_source_key(row.get("source") or row.get("text") or "")
        translation = str(row.get("translation") or "").replace("\\n", "\n").strip()
        if source and translation:
            out[source] = translation
    return out


class VevorGenerativeMergeTranslationOverrides:
    """Merge optional API translation overrides into blocks (match by source text)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blocks": ("JSON", {"forceInput": True}),
                "translation_overrides": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": '可选 JSON：{"items":[{"source":"…","translation":"…"}]}',
                    },
                ),
            }
        }

    RETURN_TYPES = ("JSON",)
    RETURN_NAMES = ("blocks",)
    FUNCTION = "merge"
    CATEGORY = "vevor/generative"

    def merge(self, blocks, translation_overrides: str = ""):
        data = [dict(b) for b in _blocks_as_list(blocks)]
        overrides = _parse_translation_override_map(translation_overrides)
        if not overrides:
            return (data,)
        for b in data:
            key = _norm_source_key(b.get("text") or "")
            if key in overrides:
                b["translation"] = overrides[key]
        return (data,)


class VevorGenerativeTranslationReport:
    """Per-block source + translation JSON for Preview / Gateway GET /jobs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blocks": ("JSON", {"forceInput": True}),
                "output_lang": (["de", "fr", "it", "es"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "build"
    CATEGORY = "vevor/generative"
    OUTPUT_NODE = True

    def build(self, blocks, output_lang: str):
        items = []
        for b in _blocks_as_list(blocks):
            if b.get("isLogo"):
                continue
            source = str(b.get("text") or "").strip()
            if not source:
                continue
            translation = str(b.get("translation") or "").strip()
            items.append(
                {
                    "id": int(b.get("id") or 0),
                    "source": source,
                    "translation": translation,
                }
            )
        payload = {"lang": output_lang, "items": items}
        report = json.dumps(payload, ensure_ascii=False, indent=2)
        return {"ui": {"text": [report]}, "result": (report,)}


class VevorGenerativeGenerateBundle:
    """Build ApiYi Sunburst image-edit prompt (design-agent ``buildTextMapPrompt``)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blocks": ("JSON", {"forceInput": True}),
                "target_lang": ("STRING", {"default": "德语"}),
                "font_dir": ("STRING", {"default": "fonts"}),
                "image_prompt_template": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "留空=内置 Sunburst 模板；占位 {{targetLang}} {{rows}}",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generate_prompt",)
    FUNCTION = "build"
    CATEGORY = "vevor/generative"

    def build(self, blocks, target_lang: str, font_dir: str, image_prompt_template: str = ""):
        bl = _blocks_as_list(blocks)
        rows = build_image_rows(bl, font_dir)
        row_text = rows if rows.strip() else '- "(no translatable rows)" -> ""'
        prompt = build_image_prompt(row_text, image_prompt_template, target_lang)
        return (prompt,)


class VevorGenerativeGlyphCard:
    """Render black reference card (demo ``lib-image.renderGlyphCard``)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "glyph_chars": ("STRING", {"forceInput": True}),
                "font_dir": ("STRING", {"default": "fonts"}),
                "font_size": ("INT", {"default": 220, "min": 32, "max": 512}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "render"
    CATEGORY = "vevor/generative"

    def render(self, glyph_chars: str, font_dir: str, font_size: int):
        arr = render_glyph_card(font_dir, glyph_chars, font_size)
        return (torch.from_numpy(arr).unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "VevorGenerativeAnalyzePrompt": VevorGenerativeAnalyzePrompt,
    "VevorGenerativeParseBlocks": VevorGenerativeParseBlocks,
    "VevorGenerativeTranslateLexicon": VevorGenerativeTranslateLexicon,
    "VevorGenerativeTranslatePrompt": VevorGenerativeTranslatePrompt,
    "VevorGenerativeApplyTranslations": VevorGenerativeApplyTranslations,
    "VevorGenerativeMergeTranslationOverrides": VevorGenerativeMergeTranslationOverrides,
    "VevorGenerativeTranslationReport": VevorGenerativeTranslationReport,
    "VevorGenerativeGenerateBundle": VevorGenerativeGenerateBundle,
    "VevorGenerativeGlyphCard": VevorGenerativeGlyphCard,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VevorGenerativeAnalyzePrompt": "Generative · Analyze Prompt",
    "VevorGenerativeParseBlocks": "Generative · Parse Blocks JSON",
    "VevorGenerativeTranslateLexicon": "Generative · Translate Lexicon",
    "VevorGenerativeTranslatePrompt": "Generative · Translate Prompt",
    "VevorGenerativeApplyTranslations": "Generative · Apply Translations",
    "VevorGenerativeMergeTranslationOverrides": "Generative · Merge Translation Overrides",
    "VevorGenerativeTranslationReport": "Generative · Translation Report",
    "VevorGenerativeGenerateBundle": "Generative · Generate Prompt (ApiYi)",
    "VevorGenerativeGlyphCard": "Generative · Glyph Reference Card",
}
