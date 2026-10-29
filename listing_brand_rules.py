# VEVOR 品牌 2.0 Listing 排版规范（26.03）— 从 PDF 提炼、供 Comfy 链路复用的约束文本。
# 源文件：VEVOR品牌2.0_Listing排版规范_26.03.pdf
# PDF 中部分色值/示意图无法 OCR；色名以「VEVOR Orange + 黑 + 白」为准，具体色值以设计稿为准。

LISTING_RULES_MARKER = "[VEVOR Listing 2.0 — layout brand rules]"

# 翻译节点 extra_rules：只约束「文案内容与层级」，字体/字号由 VevorTextRender（Inter Tight）执行。
LISTING_LAYOUT_EXTRA_RULES = """\
（VEVOR 品牌 2.0 Listing 排版规范 — 译文侧约束，与 OCR 区域一一对应）
- 品牌：VEVOR、AUVII 及型号/ SKU 保持不译、不改写；不要自造 slogan 或额外卖点。
- 每个 JSON 区域输出一段纯文本：不要换行符、不要 markdown/HTML、不要把多个区域合并成一段。
- 视觉层级：区分标题 / 副标题 / 正文 / 参数 / 图表标签；译文语气与原文同级（原文全大写标题则译文仍用对应语种的全大写或同等强调风格）。
- 篇幅：电商图空间有限——在忠实前提下优先短句；标题比正文更短；避免比英文明显更长（德语等可略长，但单区域不宜溢出原框语义）。
- M 端精品图（常见画布约 600×450）：对应 PDF 强制下限语义——标题级文案务必短、参数与数字紧凑一行；勿写长从句。
- 富文本长图（如 1600×2000、1600×1600 主图）：主次分明，核心卖点先译，次要说明可压缩但不可删规格/数字/单位。
- 色彩：品牌主色 VEVOR Orange、辅助黑/白体现在原图画面上；译文中不要写颜色名或排版指令（如「橙色标题」）。
- 禁止在文案中要求：描边、渐变、压缩拉伸字体、倾斜正文（斜体仅标题且不与正文混用——由排版节点处理，不是翻译输出）。
- 尺寸：产品尺寸以图内标注线/标注图为准；不要把尺寸再重复成纯文字列表（除非 OCR 区域本身就是尺寸句）。
- 图标/对比/功能类短标签：用目标语种的简短名词短语，便于单行排版（如对比图 Ours/Others 必须本地化，见主 prompt）。
"""

# Qwen 擦字 / 补边：追加到正向 prompt 末尾（英文，模型可读）。
LISTING_ERASE_PROMPT_SUFFIX = """\

""" + LISTING_RULES_MARKER + """
Brand listing constraints (non-text pixels):
- Preserve VEVOR logo, icons, dimension/annotation lines, QR codes, and product geometry exactly.
- Keep the brand palette (VEVOR Orange accents, black, white) in non-text areas; do not recolor or restyle the ad.
- Do not erase or blur annotation graphics, callout lines, or iconography; text removal only inside letterforms.
"""

LISTING_OUTPAINT_PROMPT_SUFFIX = """\

""" + LISTING_RULES_MARKER + """
Brand listing constraints for margin fill:
- Extend background only; keep existing ad content (product, logo, all original text) fixed in place.
- New margin pixels must match scene colour, lighting, and texture; prefer clean white/neutral margins when the original uses listing white margins.
- Do not introduce new text, logos, orange blocks, or decorative frames in filled areas.
"""

LISTING_ERASE_NEGATIVE_EXTRA = (
    "removed logo, removed icons, removed dimension lines, recolored brand orange, "
    "outlined text, gradient text, warped product"
)

# 视觉质检 typography_layout 维度补充（中文）。
LISTING_EVAL_TYPOGRAPHY_APPENDIX = (
    "   - 对照 VEVOR 品牌 2.0 Listing 排版规范：英文字体应为 Inter Tight 风格（本流程用 Inter Tight 渲染）；"
    "不得出现描边字、渐变字、被压扁/拉伸的字形；正文不宜倾斜。\n"
    "   - 品牌色 VEVOR Orange 在版面中应克制（规范建议约 20% 以内）；勿因本地化把整屏刷成杂色。\n"
    "   - M 端语义：标题/副标题/正文/参数层级是否清晰、可读；是否左对齐或居中与原图版式一致；"
    "有无信息过载、无视觉焦点、或直接等比缩放导致的糊字。\n"
    "   - 边距与截断：文字是否落在建议安全边距内（勿贴边被裁）；矩形圆角信息图是否协调（规范为参考，以原图为准）。\n"
)
