"""Large-model QA: score a rendered locale against the source ad.

Rendering successfully proves nothing about *quality*. The renderer can place
every string without overflow and still ship a German caption that means the
wrong thing, or a plate whose product was quietly repainted by the erase model.
This module turns a vision model into a reviewer and then does the arithmetic
itself, so a run produces a verdict instead of an unread paragraph.

The model is only trusted for the four dimension scores and the issue list; the
weighted total, the verdict and every threshold live in code. That split matters:
a model asked to grade itself will happily return "total: 100" next to a list of
problems, so the number is recomputed from the dimensions and flags here.
"""

import importlib.util
import json
import os

# ComfyUI loads a custom-node directory by *path*, so a plain `import
# text_render` does not resolve there. Load the sibling explicitly, the same way
# __init__.py does, instead of relying on package semantics that do not hold.
_SPEC = importlib.util.spec_from_file_location(
    "vevor_mask_union_text_render_eval", os.path.join(os.path.dirname(os.path.abspath(__file__)), "text_render.py")
)
_text_render = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_text_render)

build_name = _text_render.build_name
loads_loose = _text_render.loads_loose

# Only present inside ComfyUI. Missing here means a unit-test context, which must
# still be able to score without touching the output directory.
try:
    import folder_paths
except ImportError:  # pragma: no cover - exercised by the test harness
    folder_paths = None

# Weights sum to 1.0. Translation fidelity leads because a beautiful image with
# the wrong claim is the one failure a shopper cannot forgive; visual fidelity is
# second because "product unchanged" is the hard constraint of this pipeline.
DIMENSIONS = (
    ("translation_accuracy", 0.35, "译文准确度"),
    ("typography_layout", 0.20, "文字排版"),
    ("visual_consistency", 0.30, "画面一致性（文字以外）"),
    ("locale_conventions", 0.15, "本地化规范"),
)

DIMENSION_KEYS = tuple(key for key, _, _ in DIMENSIONS)
DIMENSION_LABELS = {key: label for key, _, label in DIMENSIONS}

# A pass needs to be good on every axis, not merely on average.
PASS_SCORE = 85.0
REVIEW_SCORE = 70.0

# Below this the picture itself is untrustworthy, whatever the average says.
VISUAL_FLOOR = 60.0

LOCALE_NAMES = {
    "de": "German (德文)",
    "fr": "French (法文)",
    "it": "Italian (意大利文)",
    "es": "Spanish (西班牙文)",
}


def build_eval_prompt(locale, render_report=""):
    """Compose the review instruction for one locale.

    ``render_report`` is the JSON the renderer emitted (font size, line count,
    overflow flags). It is passed as *evidence to verify*, never as an answer:
    the renderer knows what it intended to draw, not whether it looks right.
    """
    language = LOCALE_NAMES.get(locale, locale)
    parts = [
        f"你是亚马逊商品图片的本地化质检员。本次审查 {language}（locale={locale}）。",
        "",
        "你会收到两张图，顺序固定：",
        "  图 1 = 原图（英文广告图）",
        "  图 2 = 译文成图（本流程产出）",
        "",
        "请只在下列四个维度打分，每项 0-100 的整数，不要输出其它维度：",
        "",
        "1. translation_accuracy（译文准确度）",
        "   - 图 2 的文字是否准确传达了图 1 的语义，不缺信息、不添信息、不改变卖点。",
        "   - 产品名、型号、规格、数字、单位、品牌标识是否正确保留。",
        "   - 明显误译、漏译、把非文字元素当单词翻译（例如把 logo 碎片、把 \"0h\" 误读成 \"Oh\"）都要扣分。",
        "",
        "2. typography_layout（文字排版）",
        "   - 译文是否清晰可读，字号与原文视觉层级是否相称，没有被压扁或过度放大。",
        "   - 是否有溢出边界、压住产品、被截断、行距拥挤、文字互相重叠。",
        "   - 字重与对齐是否贴合原图风格（本流程统一使用 Inter Tight）。",
        "",
        "3. visual_consistency（画面一致性，文字以外）",
        "   - 这是硬约束：除文字区域外，产品、人物、包装、背景、配色、光影、构图必须与原图一致。",
        "   - 擦字是否留下模糊块、涂抹痕迹、接缝、色差；是否误擦掉图标、logo、二维码、装饰。",
        "   - 是否凭空新增物件，或产品形状/角度/细节被改变。",
        "",
        "4. locale_conventions（本地化规范）",
        "   - 数字与金额是否按目标语种的 CLDR 习惯书写（千位符、小数点、百分号与数字的空格）。",
        "   - 计量单位、大小写风格、标点是否自然，是否存在英文标点或英文语序残留。",
        "",
        "另外输出以下布尔标记（true/false）：",
        "  leftover_source_text —— 图 2 是否还能看到未被替换的英文原文",
        "  added_gibberish —— 是否出现乱码、重影、无意义的字符块",
        "  text_cut_off —— 是否有文字被截断或超出画布",
        "",
        "输出要求：",
        "- 只输出一个 JSON 对象，不要解释、不要 markdown 代码块。",
        "- 严格使用下面的结构：",
        '{"scores":{"translation_accuracy":0,"typography_layout":0,"visual_consistency":0,"locale_conventions":0},'
        '"flags":{"leftover_source_text":false,"added_gibberish":false,"text_cut_off":false},'
        '"issues":[{"severity":"high","dimension":"translation_accuracy","detail":"具体问题"}],'
        '"comment":"一句话结论"}',
        "- issues 只列真实观察到的问题，按严重度排序；没有问题时给空数组。",
        "- 不要输出总分，总分由流程计算。",
    ]
    report = (render_report or "").strip()
    if report:
        parts += [
            "",
            "参考：渲染器自报的排版数据（仅作核对线索，若与你在图 2 看到的冲突，以图 2 为准）：",
            report if len(report) <= 4000 else report[:4000] + "…",
        ]
    return "\n".join(parts)


def _clamp_score(value):
    """Coerce anything a model might emit into an int in 0..100."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return int(round(max(0.0, min(100.0, number))))


def parse_eval(raw):
    """Return a normalised evaluation dict, or ``None`` if unusable.

    Unusable means no parseable object with a ``scores`` mapping that yields at
    least one dimension. A partially filled answer is kept: the missing
    dimensions stay absent here, and ``aggregate_eval`` scores them as zero so a
    truncated reply cannot pass on the strength of one good axis.
    """
    payload = loads_loose(raw)
    if isinstance(payload, list):
        payload = next((item for item in payload if isinstance(item, dict) and "scores" in item), None)
    if not isinstance(payload, dict):
        return None

    scores_raw = payload.get("scores")
    if not isinstance(scores_raw, dict):
        return None
    scores = {}
    for key in DIMENSION_KEYS:
        value = _clamp_score(scores_raw.get(key))
        if value is not None:
            scores[key] = value
    if not scores:
        return None

    flags_raw = payload.get("flags")
    flags = {}
    if isinstance(flags_raw, dict):
        for key in ("leftover_source_text", "added_gibberish", "text_cut_off"):
            flags[key] = bool(flags_raw.get(key))

    issues = []
    for item in payload.get("issues") or []:
        if not isinstance(item, dict):
            continue
        detail = str(item.get("detail") or "").strip()
        if not detail:
            continue
        issues.append(
            {
                "severity": str(item.get("severity") or "medium").lower(),
                "dimension": str(item.get("dimension") or ""),
                "detail": detail,
            }
        )

    return {
        "scores": scores,
        "flags": flags,
        "issues": issues,
        "comment": str(payload.get("comment") or "").strip(),
    }


def aggregate_eval(parsed):
    """Add the weighted total and verdict to a parsed evaluation.

    The model never supplies these. Weights are renormalised over the dimensions
    The model never supplies the total; it is recomputed from the dimension
    scores, with a dimension it failed to return counted as **zero** rather than
    dropped. Dropping it (i.e. renormalising the weights) would let a truncated
    one-line answer score 100 on a single perfect axis. Hard failures also cap the
    verdict: any leftover source text or gibberish forbids a pass, and a low
    visual score caps it, because the constraint being violated is "do not change
    the ad".
    """
    scores = parsed["scores"]
    flags = parsed.get("flags") or {}
    missing = [key for key in DIMENSION_KEYS if key not in scores]

    # Weights sum to 1.0, so absent dimensions contribute nothing.
    total = round(sum(scores.get(key, 0) * weight for key, weight, _ in DIMENSIONS), 1)

    reasons = []
    if total >= PASS_SCORE:
        verdict = "pass"
    elif total >= REVIEW_SCORE:
        verdict = "review"
    else:
        verdict = "fail"

    if missing:
        reasons.append("缺少维度: " + ", ".join(missing))
    if flags.get("leftover_source_text"):
        reasons.append("仍残留英文原文")
    if flags.get("added_gibberish"):
        reasons.append("出现乱码/重影")
    if flags.get("text_cut_off"):
        reasons.append("文字被截断")

    visual = scores.get("visual_consistency")
    if visual is not None and visual < VISUAL_FLOOR:
        reasons.append(f"画面一致性 {visual} 低于下限 {int(VISUAL_FLOOR)}")

    # Any of the above means "a human should look at this", so a pass is off the
    # table; a lower verdict (fail) is left alone because it is already stricter.
    if reasons and verdict == "pass":
        verdict = "review"

    result = dict(parsed)
    result["total"] = total
    result["verdict"] = verdict
    result["penalties"] = reasons
    result["missing_dimensions"] = missing
    return result


def format_eval(result, locale):
    """Render an aggregate into the short text a human actually reads."""
    total = result.get("total")
    head = f"[{locale}] score {total if total is not None else 'n/a'}  verdict {result.get('verdict')}"
    head += f"  (pass>={int(PASS_SCORE)}, review>={int(REVIEW_SCORE)})"
    lines = [head]
    for key, weight, label in DIMENSIONS:
        value = result["scores"].get(key)
        shown = value if value is not None else "missing"
        lines.append(f"  {key:22s} {str(shown):>7s}  w={weight:.2f}  {label}")
    flags = result.get("flags") or {}
    if flags:
        lines.append("  flags: " + ", ".join(f"{k}={str(v).lower()}" for k, v in flags.items()))
    for reason in result.get("penalties") or []:
        lines.append(f"  penalty: {reason}")
    for issue in result.get("issues") or []:
        lines.append(f"  issue[{issue['severity']}] {issue['dimension']}: {issue['detail']}")
    comment = result.get("comment")
    if comment:
        lines.append(f"  comment: {comment}")
    return "\n".join(lines)


def _write_artifact(result, locale, keyword, slot, subfolder="i18n-text/eval"):
    """Persist one evaluation next to the images so a batch run stays auditable.

    Returns the relative path, or ``""`` when ComfyUI's folders are not
    importable (unit-test context) so scoring still works headlessly.
    """
    if folder_paths is None:
        return ""
    base = folder_paths.get_output_directory()
    target_dir = os.path.join(base, subfolder.replace("\\", "/"))
    os.makedirs(target_dir, exist_ok=True)
    name = build_name(keyword, locale, slot, subfolder="")
    path = os.path.join(target_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"locale": locale, "keyword": keyword, "slot": slot, **result},
                  handle, ensure_ascii=False, indent=2)
    return os.path.join(subfolder, f"{name}.json")


class VevorEvalPrompt:
    """Build the vision-model review prompt for one locale."""

    DESCRIPTION = (
        "Compose the QA rubric prompt that makes a vision model score a rendered "
        "locale against the source ad on translation, typography, non-text visual "
        "consistency and CLDR conventions."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "locale": (["de", "fr", "it", "es"], {"default": "de"}),
            },
            "optional": {
                "render_report": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build"
    CATEGORY = "vevor/eval"

    def build(self, locale, render_report=""):
        return (build_eval_prompt(locale, render_report),)


class VevorEvalScore:
    """Parse a QA answer, recompute the weighted score, and record the verdict."""

    DESCRIPTION = (
        "Turn a vision model's QA JSON into a weighted score and verdict. The "
        "total and every threshold are computed here, never taken from the model: "
        "leftover source text or gibberish forbids a pass, and a poor visual score "
        "caps it. Writes output/i18n-text/eval/<keyword>-<locale>-<slot>.json."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "locale": (["de", "fr", "it", "es"], {"default": "de"}),
                "keyword": ("STRING", {"default": "场景", "forceInput": True}),
                "slot": ("STRING", {"default": "aigc", "forceInput": True}),
                "save_json": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary",)
    FUNCTION = "score"
    CATEGORY = "vevor/eval"
    OUTPUT_NODE = True

    def score(self, raw, locale, keyword="", slot="", save_json=True):
        parsed = parse_eval(raw)
        if parsed is None:
            summary = (
                f"[{locale}] score n/a  verdict unparsed\n"
                "  模型没有返回可解析的 scores JSON。原始回复：\n"
                + (str(raw or "").strip()[:800] or "(空)")
            )
            return {"ui": {"text": [summary]}, "result": (summary,)}

        result = aggregate_eval(parsed)
        summary = format_eval(result, locale)
        if save_json:
            written = _write_artifact(result, locale, keyword, slot)
            if written:
                summary += f"\n  saved: output/{written}"
        return {"ui": {"text": [summary]}, "result": (summary,)}


NODE_CLASS_MAPPINGS = {
    "VevorEvalPrompt": VevorEvalPrompt,
    "VevorEvalScore": VevorEvalScore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VevorEvalPrompt": "Eval Prompt (QA rubric, 4 dimensions)",
    "VevorEvalScore": "Eval Score (weighted total + verdict)",
}
