"""Prompt strings aligned with ``demo/prompts.mjs`` (generative translate pipeline)."""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ANALYZE_PATH = os.path.join(_HERE, "generative_analyze_prompt.txt")


def load_analyze_prompt() -> str:
    with open(_ANALYZE_PATH, encoding="utf-8") as f:
        return f.read().strip()


def build_translate_prompt(rows: str, lang: str, extra: str = "") -> str:
    return f"""你是资深电商文案翻译专家。把下面每条文案翻译成【{lang}（面向母语消费者）】，要求：
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


def build_image_prompt(rows: str) -> str:
    return f"""TYPEFACE LOCK — every letter, digit and accent below is drawn from the second image, never from your own font.

One shape rule that matters most: every stroke ends in a flat cut parallel to the baseline — horizontal, never slanted, never rounded off.

Edit the first image: replace each text below with the new wording, keeping its anchor position and colour unchanged. Ignore the second image's layout, its background and its word order.
{rows}

Hard rules — these override everything else:
- [align:x]: every line of that block uses it — left lines share one left edge, centred lines share one centre, right lines share one right edge. Never indent, centre or shift a later line on its own.
- [size:x%]: the font size as a percentage of the image height. Fixed — never round it, never change it to make a longer text fit.
- [bold]: keep a heavy stroke heavy, and keep regular text regular.
- Keep the line breaks exactly as written: each "\\n" is one printed line break. Never hyphenate, never cut or hide a word, never split a number from its unit or a word from its punctuation.
- Never squeeze, condense, stretch, squash or slant the letters.
- If a line is wider than before, let it run into the free space beside it — never over a person, the product or another text. If there is none but there is room below, wrap whole words onto one more line.
- Keep the colours of the first image exactly: a white line stays white, an orange run stays orange together with its punctuation.
- Do not add, remove or restyle anything: no outlines, no shadows, no glow, no gradient.

Everything else in the first image stays exactly as it is — including the logos, model numbers and watermarks on the product.

Last check, before you finish — the typeface must not slip: compare every drawn word with the second image letter by letter, and if any character would not pass as the same font, redraw that word from the second image."""
