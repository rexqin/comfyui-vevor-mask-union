"""Prompt strings aligned with ``demo/prompts.mjs`` (generative translate pipeline)."""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ANALYZE_PATH = os.path.join(_HERE, "generative_analyze_prompt.txt")


def load_analyze_prompt() -> str:
    with open(_ANALYZE_PATH, encoding="utf-8") as f:
        raw = f.read().strip()
    # Recover from accidental JS export pasted into the .txt source file.
    if raw.startswith("xport") or "ANALYZE_PROMPT" in raw[:120]:
        tick = raw.find("`")
        if tick >= 0:
            raw = raw[tick + 1 :]
            end = raw.rfind("`")
            if end > 0:
                raw = raw[:end]
    return raw.strip()


def _effective_template(custom: str, default: str) -> str:
    text = (custom or "").strip()
    return text if text else default


DEFAULT_TRANSLATE_RULES = """你是资深电商文案翻译专家。把下面每条文案翻译成【{lang}（面向母语消费者）】，要求：
- 如附带原图，请结合图片理解产品与部件的实际语境，选择专业准确的术语；但只翻译清单中的条目，不要翻译图上其它文字
- 译文语气自然、符合本土表达习惯，电商短句风格，简练贴合原意，不要翻译腔
- 每条译文的行数、分行位置必须与原文一致：原文中的 \\n 表示换行，译文同样用 \\n 表示相同位置的换行；禁止把多行合并成一行，禁止跨行调换语序
- 若原文某一行只由数字/单位/型号组成（如「62cc 3.8HP 12000RPM」），译文该行也必须只含数字/单位/型号（顺序可换），不得在该行加入描述词；描述词放到对应的描述行
- 空间紧张时不要增加原文没有的冠词/介词/填充词（如 un / una / the）
- 单位与型号不翻译、不换算、不本地化：HP 保持 HP（不要写 PS），RPM 保持 RPM，cc 保持 cc
- 例外：单条原文超过 120 字符的长段落（免责声明等）整段翻译为一段即可，不要求逐行对应
- 数字、单位、型号原样保留且不加空格（如 1500W、86°F-185°F）
- 译文用于图片文字排版：短标签尽量紧凑，避免不必要地比原文显著变长
- 长段落（免责声明、法务说明、警示语）不得过度精简：译文总长度应达到原文的 85% 以上。原图留给它的位置是按原文字数留的，压得太短右侧会空出一大块；限定语、条件、警示语一条都不能省，只是换成该语言的自然说法
- 只输出一个 JSON 对象（不要 markdown 代码块、不要任何解释）：{{"编号": "译文", ...}}{extra}

条目：
{rows}"""


def build_translate_prompt(
    rows: str,
    lang: str,
    extra: str = "",
    rules_template: str = "",
) -> str:
    tpl = _effective_template(rules_template, DEFAULT_TRANSLATE_RULES)
    if "{rows}" not in tpl:
        tpl = tpl.rstrip() + "\n\n条目：\n{rows}"
    return tpl.format(lang=lang, rows=rows, extra=extra)


def build_compact_section(over_items) -> str:
    if not over_items:
        return ""
    lines = []
    for o in over_items[:5]:
        block = o["block"]
        if o.get("whole"):
            cap = f"整段不超过 {o['limit']} 字符、保持 {o['lines']} 行"
        else:
            cap = f"每行不超过 {o['limit']} 字符"
        src = block["text"].replace("\n", "⏎")
        prev = str(block.get("translation") or "").replace("\n", "⏎")
        lines.append(f"- 「{src}」→ 上一版「{prev}」超出可用空间（{cap}）")
    body = "\n".join(lines)
    return (
        "\n\n【压缩】以下条目的上一次译文超出原图可用空间（排版塞不下），本次必须压进标注的上限：\n"
        f"{body}\n"
        "先做「行内重新分配」：把超出上限的行尾词语整体移到下一行，保持原行数与断行位置，不删词、不改词；"
        "仍然放不下时，才换用更简短的同义表达或调整语序。不得丢信息点：名词、动作、数量、条件都要保留"
        "（「guide bar」不能压成「bar」）；数字、单位、型号与核心卖点词不得改动或丢弃。宁可略超上限，也不许丢信息点。"
    )


def build_paragraph_section(items) -> str:
    if not any(it.get("wholeCap") for it in items):
        return ""
    return (
        "\n\n【长段落空间适配】部分条目是整段文字（免责声明/法律行），原图给它的行数是固定的——"
        "整段译文的字符数必须压进标注总量并保持相同行数，否则生图时会裁切、缩字或压字：\n"
        "- 目标是「不超过上限」，不是越短越好：在上限内尽量保留原文的全部信息与限定语，只有超限时才逐级压缩\n"
        "- 先做行内重新分配，再换更短的同义表达（短语化、调整语序），信息点（名词/动作/数量/条件）一个都不能少\n"
        "- 数字、单位、型号原样保留\n"
        "- 不要为了凑行数把两行并成一行，也不要写成超过原行数的长段落"
    )


DEFAULT_IMAGE_PROMPT_HEADER = (
    "Edit the reference image: translate every text on it into {{targetLang}}.\n\n"
)

DEFAULT_IMAGE_PROMPT = """Everything except the glyphs themselves must stay EXACTLY as it is — no redraw, no relight, no recolor, no reframe, nothing added and nothing removed anywhere in the image. The ONE exception is the panel or plate directly behind a listed text: it may grow wider or taller to hold a longer translation, but it must never cover a person or the product, must never touch or overlap any OTHER line of text, and must never change its color or corner style. If a longer translation would run into a neighbouring line, tighten the text inside its own panel instead of expanding into it. Any panel or caption that sits flush against an image edge must stay flush — keep the same edge distance as the reference, never add a margin or gap there.

TYPOGRAPHY — this is the most important constraint, and it is a reading task, not a design choice. Re-render each text with the SAME typeface as the original, not just a similar one.
Diagnose the reference before drawing any letter. Look only at the large display text and answer two questions about it:
- The lowercase "a": is it SINGLE-STOREY (one round bowl with one straight stem on the right, no hook above) or DOUBLE-STOREY (a bowl under a curved arch with a hook on top)?
- The capital "C": is it a continuous ROUND ARC whose two ends curl toward each other leaving a small opening, or does its left side run as a FLAT VERTICAL EDGE leaving a wide opening?
Those two answers always agree and they name the face. Single-storey a + round-arc C = geometric sans. Double-storey a + flat-edged C = neo-grotesque. The "o" confirms it: near-perfect circle in the geometric case, a slightly narrow oval in the grotesque case. Read the reference and reproduce the same answer — never the one you would pick by habit.
The most common failure of this task is that the layout comes out perfect while the face is silently swapped: a geometric original is re-drawn as a neo-grotesque, because neo-grotesque is the default an image model reaches for. Inter / Roboto / Helvetica / Arial letterforms are exactly what to avoid unless the original is already one of them.
Then hold these per block:
- Keep the SAME weight: a bold line stays equally bold, a regular line stays regular. Never thin a bold line, never bolden a regular one.
- Keep the SAME letter shapes: same x-height, same stroke contrast, same character width, same tracking. Never switch to a condensed or extended width, never italicise and never slant the type. Digits keep the same width and the same numerals.
- Do NOT restyle the text: no outlines, no shadows, no gradients, no glow, no new colors, no caps/lowercase change.

COLOR — keep the exact color of every text, sampled from the original pixels:
- Use the color named for that block and keep it identical to what the original shows: never shift its hue, saturation or brightness, never add a gradient.
- Punctuation attached to a colored token (the "*" right after "40%") keeps that token's color: never split a colored token and its punctuation into two colors.
- NEVER repaint, recolor or invert any text, and never turn a colored text white or a white text colored.

LAYOUT — each line is [alignment] [color] [bold] [lines:N] [box:x,y,w,h] [panel:x,y,w,h] [safe:left/right] [group:gN] "original" -> "replacement". All numbers are percentages of the image width/height. [lines:N] is the maximum number of printed lines this text may occupy — never exceed it; use fewer only when the text genuinely fits on fewer lines. [box] is the area the text occupies in the reference, [panel] the panel or plate behind it, and [safe] the empty padding inside that panel. These tags are instructions — never print them in the image. A literal "\\n" marks a line break printed in the image; it must break the line there and must never appear in the image as characters.
{{rows}}
FONT SIZE IS FIXED — every text keeps its original font size and weight. Never shrink or enlarge a line to make a longer translation fit.

WORD AND LETTER SPACING — spacing is natural and uniform from the first character to the last. Never add extra space between words or around numbers, never justify a line, and never keep a translated word in the x-position of the original word: word order may change, so lay the replacement out with even word spacing inside its box. Only the box alignment anchor, the baseline, the font size and the weight stay fixed.

WORD ORDER — render the words of every replacement in the exact order given, left to right. Never move a word, number or symbol to the line start or to another position just because the original line started with it; the quoted replacement is the authoritative text.

SHORT TEXT — a title, spec line or label: keep exactly the printed line breaks and the [lines:N] line count. Never wrap it onto an extra line, never split a number from its unit, and never redistribute its spaces. If it is slightly wider than the original, let it extend into free space around its box without moving its left anchor and without distorting any character.

LONG PARAGRAPH — a legal line or disclaimer: keep the [lines:N] line count, keep the same left inset on every line and keep the [safe] padding empty. Never cut, clip or hide a word. If the panel behind it has free room above, the panel may grow upward by at most one line, but it must never cover another text, a person or the product. Never shrink the font to force it into fewer lines.

OVERFLOW — if a line still does not fit, move whole words to the next line within the allowed [lines:N]; only a long paragraph may use one extra line, and only when its panel has free room. Never shorten or paraphrase the given text, never stretch word gaps, never squeeze characters and never let any word touch or cross the right edge of its box or panel.

Keep the same alignment on every line, the same weight and the same typeface. Copy every replacement EXACTLY as given — never add, drop or paraphrase words. Blocks carrying the same [group:gN] tag form one set and must look identical — same font size, same panel size, same line count, same letterforms; if one of them grows, the whole set grows the same way, and never leave one member looking different from its siblings. Do NOT translate, redraw or remove any text that is not listed above — logos, model numbers, watermarks and legal lines are not text here and must stay exactly as they are.

FINAL SELF-CHECK before you return the image — compare your own output against the reference: if the reference's lowercase "a" has no hook on top and your "a" does, or the reference's "C" is a round arc and yours has a flat vertical left edge, then you have broken the typeface constraint: redraw that text with the reference's letterforms."""


def build_image_prompt(rows: str, header_template: str = "", target_lang: str = "") -> str:
    custom = (header_template or "").strip()
    if custom:
        tpl = custom
    else:
        tpl = DEFAULT_IMAGE_PROMPT_HEADER + DEFAULT_IMAGE_PROMPT
    lang = (target_lang or "").strip() or "target language"
    out = tpl.replace("{{targetLang}}", lang).replace("{{rows}}", rows)
    if "{rows}" in out and rows not in out:
        out = out.format(rows=rows, targetLang=lang)
    return out
