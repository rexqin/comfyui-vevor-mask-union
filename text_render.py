"""Deterministic text rendering for translated ad copy.

Why this exists
---------------
A diffusion model cannot be told "use Inter Tight SemiBold, 42 px, fit these
words into a 380 px box, wrap to at most 3 lines". It paints plausible-looking
glyphs, so every language comes out in a different typeface, long translations
overflow their box, and nothing about the result is reproducible. That is why
the translation pipeline needs the text to be *drawn*, not *denoised*.

The geometry is already available: ``Apply EasyOCR`` emits labelme JSON whose
shapes carry pixel bounding boxes and the recognised source string. That JSON is
the contract here -- this node consumes it, looks up a translation per region,
and renders that translation inside the original box with a real font file,
shrinking the point size and wrapping lines until it fits.

This node deliberately does NOT translate. Translation is a separate concern (an
LLM call or a prepared mapping); keeping it out of the node makes the render
deterministic and testable without a network.

The companion problem -- the original glyphs are still painted in the pixels --
is handled upstream: the diffusion model is used only to *erase* the text, and
its output is composited back through the OCR mask so non-text pixels are
untouched.
"""

import json
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:  # ComfyUI always provides torch; absent only when running unit tests.
    import torch
except Exception:  # pragma: no cover
    torch = None

# Inter Tight ships one static file per weight; the key is also what ComfyUI
# shows in the weight dropdown.
WEIGHT_FILES = {
    "Thin": "InterTight-Thin.ttf",
    "ExtraLight": "InterTight-ExtraLight.ttf",
    "Light": "InterTight-Light.ttf",
    "Regular": "InterTight-Regular.ttf",
    "Medium": "InterTight-Medium.ttf",
    "SemiBold": "InterTight-SemiBold.ttf",
    "Bold": "InterTight-Bold.ttf",
    "ExtraBold": "InterTight-ExtraBold.ttf",
    "Black": "InterTight-Black.ttf",
}

# CLDR number conventions for the locales this pipeline emits. The group
# separator matters: French uses a narrow no-break space, the others a full stop.
LOCALE_SEPARATORS = {
    "de": {"group": ".", "decimal": ","},
    "fr": {"group": "\u202f", "decimal": ","},
    "it": {"group": ".", "decimal": ","},
    "es": {"group": ".", "decimal": ","},
    "en": {"group": ",", "decimal": "."},
}

# A number that already carries a group separator or a two-digit decimal tail.
# Bare integers are left alone: "1200" is a model number as often as a quantity,
# and rewriting the punctuation of a part number would be a defect, not a fix.
_NUMBER_RE = re.compile(r"\d{1,3}(?:[.,\u00a0\u202f]\d{3})+(?:[.,]\d+)?|\d+[.,]\d{2}(?!\d)")


def _norm_number(token, separators):
    """Re-punctuate one numeric token into the target locale.

    The token is parsed structurally instead of trusting its punctuation: the
    last separator followed by exactly two digits is the decimal mark, anything
    else is grouping. That is what makes "1,200.00", "1.200,00" and "1 200,00"
    all normalise to the same German output.
    """
    decimals = ""
    head = token.replace("\u00a0", "").replace("\u202f", "")
    m = re.search(r"[.,](\d{2})$", head)
    if m:
        decimals = m.group(1)
        head = head[: m.start()]
    digits = re.sub(r"\D", "", head)
    if not digits:
        return token
    groups = []
    while len(digits) > 3:
        groups.insert(0, digits[-3:])
        digits = digits[:-3]
    groups.insert(0, digits)
    out = separators["group"].join(groups)
    if decimals:
        out += separators["decimal"] + decimals
    return out


def format_numbers(text, locale):
    """Apply CLDR digit grouping and decimal marks to one string."""
    separators = LOCALE_SEPARATORS.get(locale)
    if not separators:
        return text
    return _NUMBER_RE.sub(lambda m: _norm_number(m.group(0), separators), text)


def _iter_labelme(payload):
    """Yield labelme dicts from whatever shape the JSON arrived in.

    ``Apply EasyOCR`` returns a *list* of labelme dicts (one per input frame).
    ComfyUI may hand that over as the list itself or as its serialised string,
    and a single-image run can also arrive as one bare dict.
    """
    if payload is None:
        return
    if isinstance(payload, str):
        payload = payload.strip()
        if not payload:
            return
        payload = json.loads(payload)
    if isinstance(payload, dict):
        if "shapes" in payload:
            yield payload
            return
        inner = payload.get("result")
        if isinstance(inner, dict):
            for doc in _iter_labelme(inner):
                yield doc
        return
    if isinstance(payload, (list, tuple)):
        for item in payload:
            for doc in _iter_labelme(item):
                yield doc


def parse_ocr(payload, with_index=False):
    """Return ``(regions, canvas_w, canvas_h)`` from an EasyOCR JSON payload.

    Each region is ``{"box": (x1, y1, x2, y2), "text": str, "score": float}`` in
    pixels on the OCR canvas. Only the first labelme document is used: a batch
    would mean several source images, and one render pass cannot address those.

    ``with_index`` keeps the region's own ``index`` label when the payload came
    from :class:`VevorOcrGuard` instead of EasyOCR. The guard removes regions, so
    position and index diverge -- the translation prompt keys answers by index,
    and the renderer must look them up by the same number.
    """
    for doc in _iter_labelme(payload):
        width = int(doc.get("imageWidth") or 0)
        height = int(doc.get("imageHeight") or 0)
        regions = []
        for position, shape in enumerate(doc.get("shapes") or []):
            points = shape.get("points") or []
            if len(points) < 2:
                continue
            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]
            label = shape.get("label") or ""
            score = 0.0
            # EasyOCR labels each region "<text>:<confidence>".
            head, sep, tail = label.rpartition(":")
            if sep:
                try:
                    score = float(tail)
                    label = head
                except ValueError:
                    pass
            if not str(label).strip():
                continue
            region = {
                "box": (min(xs), min(ys), max(xs), max(ys)),
                "text": str(label),
                "score": score,
            }
            if with_index:
                raw = shape.get("index")
                region["index"] = raw if isinstance(raw, int) else position
                if shape.get("literal"):
                    region["literal"] = True
            regions.append(region)
        return regions, width, height
    return [], 0, 0


def loads_loose(payload):
    """Parse JSON that a language model may have wrapped in prose or a fence.

    LLMs habitually answer with ```json ... ``` or a sentence before the object.
    Both the translation map and the evaluation scores arrive that way, so the
    tolerance lives here instead of being re-implemented per consumer.
    """
    if payload is None:
        return None
    if not isinstance(payload, str):
        return payload
    text = payload.strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith(("{", "[")):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_translations(payload):
    """Return a lookup mapping region key -> translated string.

    Accepted keys, checked per region in this order: the region index (as string
    or int) and then the original source text. Index keys survive a source string
    that appears twice ("SALE", "SALE"), which text keys cannot distinguish.
    """
    if payload is None:
        return {}
    if isinstance(payload, str):
        payload = loads_loose(payload)
        if payload is None:
            return {}
    if isinstance(payload, dict):
        inner = payload.get("translations")
        if isinstance(inner, (dict, list)):
            payload = inner
        if isinstance(payload, dict):
            return {str(k): str(v) for k, v in payload.items()}
    if isinstance(payload, list):
        out = {}
        for item in payload:
            if not isinstance(item, dict) or "translation" not in item:
                continue
            key = item.get("index", item.get("text"))
            if key is not None:
                out[str(key)] = str(item["translation"])
        return out
    return {}


def _resolve_font_dir(user_dir):
    """Find the directory holding the Inter Tight files.

    Several locations are tried so the pack works from its own directory as well
    as from a models/ or input/ drop, which is how a font usually reaches a
    container that cannot be rebuilt.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if user_dir:
        candidates.append(user_dir)
        candidates.append(os.path.join(here, user_dir))
    candidates.append(os.path.join(here, "fonts"))
    try:
        import folder_paths

        models_dir = folder_paths.models_dir
        candidates.append(os.path.join(models_dir, "fonts"))
        candidates.append(os.path.join(os.path.dirname(models_dir), "input", "fonts"))
    except Exception:
        pass
    for path in candidates:
        if path and os.path.isdir(path):
            try:
                names = os.listdir(path)
            except OSError:
                continue
            if any(n.lower().endswith(".ttf") for n in names):
                return path
    return None


def _load_font(font_dir, weight, size):
    name = WEIGHT_FILES.get(weight, WEIGHT_FILES["Regular"])
    size = max(1, int(size))
    for path in (
        os.path.join(font_dir, name) if font_dir else None,
        os.path.join(font_dir, WEIGHT_FILES["Regular"]) if font_dir else None,
    ):
        if not path:
            continue
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    # Last resort: the bitmap fallback still puts text in the box, which beats
    # aborting a 4-language run for one missing weight file.
    return ImageFont.load_default()


def _text_width(font, text, spacing=0.0):
    if not text:
        return 0.0
    if not spacing:
        return float(font.getlength(text))
    return float(sum(font.getlength(ch) for ch in text) + spacing * (len(text) - 1))


def _wrap(text, font, max_w, spacing):
    """Greedy word wrap; hard-split any single word wider than the whole box."""
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if _text_width(font, candidate, spacing) <= max_w:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        while _text_width(font, word, spacing) > max_w:
            cut = len(word)
            while cut > 1 and _text_width(font, word[:cut], spacing) > max_w:
                cut -= 1
            lines.append(word[:cut])
            word = word[cut:]
        current = word
    if current:
        lines.append(current)
    return lines


def _line_height(font):
    box = font.getbbox("Ag")
    return float(box[3] - box[1])


def _fit(text, font_dir, weight, box_w, box_h, max_lines, min_px, line_spacing, spacing):
    """Largest point size at which the text fits the box within ``max_lines``.

    Binary search rather than a linear walk: glyph metrics are monotone in size,
    so the "fits" predicate flips exactly once. Returns
    ``(size, lines, line_h, step, truncated)``.
    """
    ceiling = max(int(min_px) + 1, int(box_h * 1.35))
    best = None
    lo, hi = int(min_px), ceiling
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _load_font(font_dir, weight, mid)
        lines = _wrap(text, font, box_w, spacing)
        line_h = _line_height(font)
        step = line_h * line_spacing
        widest = max((_text_width(font, ln, spacing) for ln in lines), default=0.0)
        if len(lines) <= max_lines and step * len(lines) <= box_h * 1.02 and widest <= box_w * 1.02:
            best = (mid, lines, line_h, step, False)
            lo = mid + 1
        else:
            hi = mid - 1
    if best:
        return best
    # Nothing fit: emit the smallest size and let the caller report truncation
    # rather than silently dropping the region.
    font = _load_font(font_dir, weight, min_px)
    lines = _wrap(text, font, box_w, spacing)
    line_h = _line_height(font)
    return (int(min_px), lines[:max_lines], line_h, line_h * line_spacing, len(lines) > max_lines)


def _sample_text_color(source, box):
    """Pick the colour the original glyphs were drawn in.

    Text is a minority of pixels in its box, so the two extremes of the luminance
    histogram are background and ink. Whichever extreme sits further from the
    median is the ink.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1 = max(0, min(x1, source.width - 1))
    y1 = max(0, min(y1, source.height - 1))
    x2 = max(x1 + 1, min(x2, source.width))
    y2 = max(y1 + 1, min(y2, source.height))
    crop = np.asarray(source.crop((x1, y1, x2, y2)).convert("RGB")).reshape(-1, 3)
    if crop.size == 0:
        return (0, 0, 0)
    luma = crop @ np.array([0.299, 0.587, 0.114])
    median = float(np.median(luma))
    dark = float(np.percentile(luma, 2))
    light = float(np.percentile(luma, 98))
    target = dark if (median - dark) >= (light - median) else light
    spread = float(np.std(luma))
    near = np.abs(luma - target) <= max(1.0, spread * 0.5)
    if not near.any():
        idx = int(np.argmin(np.abs(luma - target)))
        return tuple(int(v) for v in crop[idx])
    return tuple(int(v) for v in np.median(crop[near], axis=0))


def _parse_color(value, fallback=(0, 0, 0)):
    if not value:
        return fallback
    value = value.strip()
    if value.startswith("#"):
        hexpart = value[1:]
        if len(hexpart) == 3:
            hexpart = "".join(c * 2 for c in hexpart)
        if len(hexpart) >= 6:
            try:
                return tuple(int(hexpart[i : i + 2], 16) for i in (0, 2, 4))
            except ValueError:
                return fallback
    m = re.fullmatch(r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", value)
    if m:
        return tuple(int(v) for v in m.groups())
    return fallback


def render_regions(plate, source, ocr_json, translations, locale, font_dir, font_weight,
                   max_lines, min_font_px, line_spacing, letter_spacing, halign, valign,
                   color_mode, custom_color, cldr_numbers, box_shrink,
                   preserve_numbers=True):
    """Draw every translated region onto ``plate`` and describe what happened.

    Kept free of torch so it can be exercised directly from unit tests.
    """
    regions, canvas_w, canvas_h = parse_ocr(ocr_json, with_index=True)
    lookup = parse_translations(translations)

    # The OCR ran on its own canvas (usually the upscaled plate); map boxes onto
    # whatever size the plate passed in happens to be.
    scale_x = plate.width / canvas_w if canvas_w else 1.0
    scale_y = plate.height / canvas_h if canvas_h else 1.0

    draw = ImageDraw.Draw(plate)
    report = []
    for position, region in enumerate(regions):
        index = region.get("index", position)
        translated = lookup.get(str(index))
        if translated is None:
            translated = lookup.get(region["text"])
        if not translated and region.get("literal"):
            # A spec cell the guard kept away from the translator: the OCR text
            # *is* the final text, so copy it rather than skipping the region.
            translated = region["text"]
        if not translated:
            report.append({"index": index, "source": region["text"], "skipped": "no translation"})
            continue
        model_translation = translated
        if cldr_numbers:
            translated = format_numbers(translated, locale)
        if preserve_numbers and not numbers_preserved(region["text"], translated):
            # A rewritten spec is always wrong; the OCR text is the best evidence
            # available, so fall back to it and say so instead of shipping 38 HP.
            translated = region["text"]
            model_translation = None
            fallback = "number missing"
        else:
            fallback = None

        x1, y1, x2, y2 = region["box"]
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y
        box_w = max(1.0, (x2 - x1) * box_shrink)
        box_h = max(1.0, (y2 - y1) * box_shrink)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if color_mode == "sample_from_source":
            color = _sample_text_color(source, (x1, y1, x2, y2))
        else:
            color = _parse_color(custom_color)

        size, lines, line_h, step, truncated = _fit(
            translated, font_dir, font_weight, box_w, box_h, max_lines,
            min_font_px, line_spacing, letter_spacing,
        )
        font = _load_font(font_dir, font_weight, size)
        total_h = step * len(lines)
        if valign == "center":
            top = cy - total_h / 2.0
        elif valign == "top":
            top = y1
        else:
            top = y2 - total_h

        widest = 0.0
        for line_no, line in enumerate(lines):
            width = _text_width(font, line, letter_spacing)
            widest = max(widest, width)
            if halign == "center":
                left = cx - width / 2.0
            elif halign == "left":
                left = x1
            else:
                left = x2 - width
            y = top + line_no * step
            if letter_spacing:
                cursor = left
                for ch in line:
                    draw.text((cursor, y), ch, font=font, fill=color)
                    cursor += font.getlength(ch) + letter_spacing
            else:
                draw.text((left, y), line, font=font, fill=color)

        report.append(
            {
                "index": index,
                "source": region["text"],
                "translation": translated,
                "font_px": size,
                "lines": len(lines),
                "widest_px": round(widest, 1),
                "box_w_px": round(box_w, 1),
                "overflows_box": widest > box_w * 1.02,
                "truncated": truncated,
                "color": list(color),
                **({"literal": True} if region.get("literal") else {}),
                **({"fallback": fallback, "model_translation": model_translation}
                   if fallback else {}),
            }
        )
    return plate, report


class VevorTextRender:
    """Render translations inside the OCR boxes of a cleared plate."""

    DESCRIPTION = (
        "Draw translations into OCR bounding boxes with a real font file, "
        "auto-fitting point size and wrapping lines so long locales do not "
        "overflow. Consumes Apply EasyOCR's JSON output."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # Apply EasyOCR declares its JSON output as type "JSON" (a list of
                # labelme docs once it reaches Python), so this must not be STRING
                # or ComfyUI rejects the link before the node ever runs.
                "ocr_json": ("JSON", {"forceInput": True}),
                "translations": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "locale": (["de", "fr", "it", "es", "en"], {"default": "de"}),
                "font_dir": ("STRING", {"default": "fonts"}),
                "font_weight": (list(WEIGHT_FILES.keys()), {"default": "SemiBold"}),
                "max_lines": ("INT", {"default": 3, "min": 1, "max": 12}),
                "min_font_px": ("INT", {"default": 10, "min": 4, "max": 200}),
                "line_spacing": ("FLOAT", {"default": 1.12, "min": 0.8, "max": 3.0, "step": 0.01}),
                "letter_spacing": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 20.0, "step": 0.1}),
                "halign": (["center", "left", "right"], {"default": "center"}),
                "valign": (["center", "top", "bottom"], {"default": "center"}),
                "color_mode": (["sample_from_source", "custom"], {"default": "sample_from_source"}),
                "custom_color": ("STRING", {"default": "#111111"}),
                "cldr_numbers": ("BOOLEAN", {"default": False}),
                "box_shrink": ("FLOAT", {"default": 0.96, "min": 0.5, "max": 1.0, "step": 0.01}),
                "preserve_numbers": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "source_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "report")
    FUNCTION = "render"
    CATEGORY = "vevor/text"

    def render(self, image, ocr_json, translations, locale, font_dir, font_weight,
               max_lines, min_font_px, line_spacing, letter_spacing, halign, valign,
               color_mode, custom_color, cldr_numbers, box_shrink, preserve_numbers=True,
               source_image=None):
        resolved_font_dir = _resolve_font_dir(font_dir)

        # Work on the first frame: a batch here would mean multiple plates, and
        # one OCR document describes exactly one of them.
        frame = image[0] if image.dim() == 4 else image
        array = np.clip(255.0 * frame.cpu().numpy(), 0, 255).astype(np.uint8)
        plate = Image.fromarray(array).convert("RGB")

        source = plate
        if source_image is not None:
            src_frame = source_image[0] if source_image.dim() == 4 else source_image
            src_array = np.clip(255.0 * src_frame.cpu().numpy(), 0, 255).astype(np.uint8)
            source = Image.fromarray(src_array).convert("RGB")

        plate, report = render_regions(
            plate, source, ocr_json, translations, locale, resolved_font_dir,
            font_weight, max_lines, min_font_px, line_spacing, letter_spacing,
            halign, valign, color_mode, custom_color, cldr_numbers, box_shrink,
            preserve_numbers,
        )

        out = np.asarray(plate).astype(np.float32) / 255.0
        return (torch.from_numpy(out).unsqueeze(0), json.dumps(report, ensure_ascii=False, indent=1))


def _align_inset(inner: int, outer: int, align: str) -> int:
    slack = outer - inner
    if align in ("top", "left"):
        return 0
    if align in ("bottom", "right"):
        return max(0, slack)
    return max(0, slack // 2)


def letterbox_to_target(
    array,
    target_w,
    target_h,
    fill_mode,
    fill_color,
    valign="center",
    halign="center",
):
    """Fit inside target_w×target_h, place content, return (rgb float32, margin mask float32).

    The margin mask is 1.0 on synthetic border pixels (candidates for diffusion
    outpaint) and 0.0 on the scaled original content.
    """
    src = Image.fromarray(array).convert("RGB")
    if src.width <= 0 or src.height <= 0:
        raise ValueError("letterbox_to_target: empty image")

    scale = min(target_w / src.width, target_h / src.height)
    new_w = max(1, int(round(src.width * scale)))
    new_h = max(1, int(round(src.height * scale)))
    scaled = src.resize((new_w, new_h), Image.LANCZOS)
    left = _align_inset(new_w, target_w, halign)
    top = _align_inset(new_h, target_h, valign)

    margin_mask = np.ones((target_h, target_w), dtype=np.float32)
    margin_mask[top : top + new_h, left : left + new_w] = 0.0

    if fill_mode == "solid":
        canvas = Image.new("RGB", (target_w, target_h), _parse_color(fill_color, (255, 255, 255)))
        canvas.paste(scaled, (left, top))
    elif fill_mode == "reflect":
        data = np.asarray(scaled)
        pad_top = top
        pad_bottom = target_h - new_h - top
        pad_left = left
        pad_right = target_w - new_w - left
        padded = np.pad(
            data,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="reflect",
        )
        canvas = Image.fromarray(padded)
    else:
        data = np.asarray(scaled)
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        padded[top : top + new_h, left : left + new_w] = data
        if left:
            padded[top : top + new_h, :left] = data[:, :1]
        if target_w - new_w - left:
            padded[top : top + new_h, left + new_w :] = data[:, -1:]
        if top:
            padded[:top, :] = padded[top : top + 1, :]
        if target_h - new_h - top:
            padded[top + new_h :, :] = padded[top + new_h - 1 : top + new_h, :]
        canvas = Image.fromarray(padded)

    rgb = np.asarray(canvas).astype(np.float32) / 255.0
    return rgb, margin_mask


class VevorLetterboxToTarget:
    """Letterbox upscaled art to a target resolution and expose a margin mask.

    Downstream diffusion (Qwen Image Edit) should fill mask==1 areas only, via
    ``ImageCompositeMasked``, so the product pixels stay untouched.
    """

    DESCRIPTION = (
        "Scale to fit inside target_w×target_h, align on canvas (e.g. bottom for "
        "footer text flush with the plate bottom), output margin mask for outpaint."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_w": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 8}),
                "target_h": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 8}),
                "fill_mode": (["reflect", "edge_extend", "solid"], {"default": "reflect"}),
                "fill_color": ("STRING", {"default": "#ffffff"}),
                "valign": (["center", "top", "bottom"], {"default": "bottom"}),
                "halign": (["center", "left", "right"], {"default": "center"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "margin_mask")
    FUNCTION = "letterbox"
    CATEGORY = "vevor/image"

    def letterbox(self, image, target_w, target_h, fill_mode, fill_color, valign, halign):
        frames = []
        masks = []
        for frame in image:
            array = np.clip(255.0 * frame.cpu().numpy(), 0, 255).astype(np.uint8)
            rgb, margin = letterbox_to_target(
                array, target_w, target_h, fill_mode, fill_color, valign, halign
            )
            frames.append(rgb)
            masks.append(margin)
        return (
            torch.from_numpy(np.stack(frames).astype(np.float32)),
            torch.from_numpy(np.stack(masks).astype(np.float32)),
        )


class VevorPadSquare:
    """Pad to a square canvas without cropping, then resize to a target side.

    Amazon listing images are square while the source ads are not. Cropping to
    reach 1:1 would move the product, so canvas is added instead and the original
    content stays centred.
    """

    DESCRIPTION = (
        "Centre an image on a square canvas (edge-extend or solid fill) and "
        "resize to a target side length. Never crops."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_size": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 8}),
                "fill_mode": (["edge_extend", "solid"], {"default": "edge_extend"}),
                "fill_color": ("STRING", {"default": "#ffffff"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pad"
    CATEGORY = "vevor/image"

    def pad(self, image, target_size, fill_mode, fill_color):
        frames = []
        for frame in image:
            array = np.clip(255.0 * frame.cpu().numpy(), 0, 255).astype(np.uint8)
            frames.append(pad_to_square(array, target_size, fill_mode, fill_color))
        # pad_to_square already returns 0..1 floats. Dividing by 255 again here
        # would crush every pixel to <= 1/255 and save an all-black image.
        return (torch.from_numpy(np.stack(frames).astype(np.float32)),)


def pad_to_square(array, target_size, fill_mode, fill_color):
    """Square-pad one uint8 HxWx3 array and resize it. Torch-free for testing."""
    src = Image.fromarray(array).convert("RGB")
    side = max(src.width, src.height)
    left = (side - src.width) // 2
    top = (side - src.height) // 2

    if fill_mode == "solid":
        canvas = Image.new("RGB", (side, side), _parse_color(fill_color, (255, 255, 255)))
        canvas.paste(src, (left, top))
    else:
        # Replicate the outermost row and column outward. For a product shot on a
        # flat backdrop this is seamless, whereas stretching a 1 px strip across
        # the whole margin would smear a gradient.
        data = np.asarray(src)
        padded = np.zeros((side, side, 3), dtype=np.uint8)
        padded[top : top + src.height, left : left + src.width] = data
        if left:
            padded[top : top + src.height, :left] = data[:, :1]
        if side - src.width - left:
            right = side - src.width - left
            padded[top : top + src.height, left + src.width :] = data[:, -1:]
        if top:
            padded[:top, :] = padded[top : top + 1, :]
        if side - src.height - top:
            padded[top + src.height :, :] = padded[top + src.height - 1 : top + src.height, :]
        canvas = Image.fromarray(padded)

    canvas = canvas.resize((target_size, target_size), Image.LANCZOS)
    return np.asarray(canvas).astype(np.float32) / 255.0


LOCALE_NAMES = {
    "de": "German (德文)",
    "fr": "French (法文)",
    "it": "Italian (意大利文)",
    "es": "Spanish (西班牙文)",
}


def _ocr_as_text(ocr_json):
    """Serialise the OCR payload for embedding in the translation prompt.

    ``Apply EasyOCR`` hands over a live Python list once the link is typed as
    JSON, but a hand-pasted payload is still a string; normalise both so the
    prompt always shows the model readable JSON rather than ``repr`` output.
    """
    if ocr_json is None:
        return ""
    if isinstance(ocr_json, str):
        return ocr_json
    return json.dumps(ocr_json, ensure_ascii=False, indent=1)


# Editable on the Comfy node; {language} / {locale} are substituted at runtime.
TRANSLATION_PROMPT_BODY_TEMPLATE = (
    "你是亚马逊商品图片的本地化文案专家。下面是 OCR 识别出的原文区域清单（JSON）。\n"
    "请把每个区域的原文翻译成 {language}。\n"
    "输出要求：\n"
    "- 只输出一个 JSON 对象，不要解释、不要 markdown 代码块。\n"
    '- 键是区域下标字符串（如 "0"、"3"），值是译文。\n'
    "- 覆盖所有区域，不得遗漏，不得合并或拆分区域。\n"
    "翻译要求：\n"
    "- 忠实传达原意，不增删信息，不改变产品名、型号、规格与数字。\n"
    "- 遵循 CLDR/ICU 规范：数字与金额按 {locale} 的千位分隔符和小数点书写，"
    "货币与日期格式本地化，注意复数与阴阳性一致。\n"
    "- 电商图片空间有限：在保住含义的前提下用简洁表达，优先短词，避免冗长从句。\n"
    "- 沿用原文的视觉风格：全大写标题仍保持全大写，简洁短句仍保持短句。\n"
    "- 品牌名、型号、条码等专有标识保持原样，不要翻译或改写。\n"
    "- 数字、单位、型号必须保持原值：可以按目标语种更换千位符/小数点，"
    "但不得改变数字本身，也不要替原文\"纠正\"数值。\n"
    "- 若某个区域像是被 OCR 切碎的字形碎片（如 \"Vow\"、\"@0\"），"
    "按原样返回，不要猜测它的含义，更不要意译。\n"
    '- 对比图/图表标签必须本地化："Ours"、"Others" 等短标签要译成目标语'
    "（德语 Unsere/Wir、Andere；法语 Notre/Nos、Autres；意大利语 Nostri、Altri；"
    "西班牙语 Nuestros/Nuestra、Otros），不要保留英文。"
)


def _substitute_locale_placeholders(body, locale):
    language = LOCALE_NAMES.get(locale, locale)
    return body.replace("{language}", language).replace("{locale}", locale)


def _translation_instruction_body(locale, prompt_body="", extra_rules=""):
    """Instruction text only (no OCR appendix)."""
    raw = (prompt_body or "").strip()
    if raw:
        body = _substitute_locale_placeholders(raw, locale)
    else:
        body = _substitute_locale_placeholders(TRANSLATION_PROMPT_BODY_TEMPLATE, locale)
    rules = (extra_rules or "").strip()
    if rules:
        return body + f"\n\n额外要求：\n{rules}"
    return body


def build_translation_prompt(ocr_json, locale, extra_rules="", prompt_body=""):
    """Compose the instruction that turns OCR regions into a translation map.

    The output is keyed by region index on purpose: a phrase repeated in the ad
    (``SALE`` twice) is two independent translations, which a source-text key
    could not tell apart. The locale's CLDR conventions are spelled out because
    a model left to itself will keep US punctuation for German prices.
    """
    parts = [
        _translation_instruction_body(locale, prompt_body, extra_rules),
        "OCR 原文：\n" + _ocr_as_text(ocr_json),
    ]
    return "\n".join(parts)


class VevorTranslationPrompt:
    """Build the translation prompt from OCR output for one locale."""

    DESCRIPTION = (
        "Compose the JSON-in/JSON-out translation instruction for one locale "
        "from Apply EasyOCR's JSON. The multiline prompt widget is shown on the "
        "node and can be edited; OCR JSON is appended when the graph runs."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ocr_json": ("JSON", {"forceInput": True}),
                "locale": (["de", "fr", "it", "es"], {"default": "de"}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": TRANSLATION_PROMPT_BODY_TEMPLATE,
                    },
                ),
                "extra_rules": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build"
    CATEGORY = "vevor/text"
    OUTPUT_NODE = True

    def build(self, ocr_json, locale, prompt="", extra_rules=""):
        full = build_translation_prompt(ocr_json, locale, extra_rules, prompt_body=prompt)
        preview = full if len(full) <= 6000 else full[:6000] + "\n…"
        return {"ui": {"text": [preview]}, "result": (full,)}


class VevorSaveNamed:
    """Save images under a *connected* filename prefix.

    Core ``SaveImage`` takes its prefix as a widget, so it cannot be driven by the
    naming node; that is the whole reason this exists. Prefix may contain ``/``,
    which becomes a subfolder under the output directory.
    """

    DESCRIPTION = (
        "Save images to output/<prefix>_#####_.png with the prefix supplied by a "
        "connection, so keyword-locale-slot naming can be computed upstream."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "vevor/image"

    def save(self, images, filename_prefix):
        import folder_paths

        output_dir = folder_paths.get_output_directory()
        prefix = (filename_prefix or "image").replace("\\", "/").strip("/")
        subfolder = os.path.dirname(prefix)
        base = os.path.basename(prefix) or "image"
        target_dir = os.path.join(output_dir, subfolder) if subfolder else output_dir
        os.makedirs(target_dir, exist_ok=True)

        existing = set(os.listdir(target_dir))
        counter = 1
        while f"{base}_{counter:05d}_.png" in existing:
            counter += 1

        saved = []
        for offset, frame in enumerate(images):
            array = np.clip(255.0 * frame.cpu().numpy(), 0, 255).astype(np.uint8)
            name = f"{base}_{counter + offset:05d}_.png"
            Image.fromarray(array).convert("RGB").save(os.path.join(target_dir, name))
            saved.append(os.path.join(subfolder, name) if subfolder else name)
        return {"ui": {"images": [{"filename": os.path.basename(p), "subfolder": subfolder, "type": "output"} for p in saved]}}


class VevorNameOutput:
    """Build the ``<keyword>-<locale>-<slot>`` output name.

    The workspace naming rule works from a source-market keyword that sometimes
    carries a ``-us`` suffix which must be dropped before the destination locale
    is appended. That branch cannot be expressed in a string template, so the
    rule lives in a node.
    """

    DESCRIPTION = (
        "Compose the listing image filename: keyword-locale-slot, dropping a "
        "trailing -us from the keyword first."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keyword": ("STRING", {"default": "场景", "forceInput": True}),
                "locale": (["de", "fr", "it", "es"], {"default": "de"}),
                "slot": ("STRING", {"default": "aigc", "forceInput": True}),
                "subfolder": ("STRING", {"default": "i18n-text"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename_prefix",)
    FUNCTION = "build"
    CATEGORY = "vevor/text"

    def build(self, keyword, locale, slot, subfolder):
        return (build_name(keyword, locale, slot, subfolder),)


def build_name(keyword, locale, slot, subfolder="i18n-text"):
    """``keyword-locale-slot``, with only a *delimited* ``-us`` tail stripped.

    A bare trailing "us" is left alone: "scenus" must not become "scen".
    """
    kw = (keyword or "").strip().strip("-")
    for tail in ("-us", "_us"):
        if kw.lower().endswith(tail) and len(kw) > len(tail):
            kw = kw[: -len(tail)].strip("-").strip()
            break
    parts = [p for p in (kw, locale, (slot or "").strip()) if p]
    name = "-".join(parts)
    folder = (subfolder or "").strip().strip("/")
    return f"{folder}/{name}" if folder else name


# ---------------------------------------------------------------- OCR guards
#
# EasyOCR is the weakest link here. On a dense ad it reads 62CC as G2CC, 12000RPM
# as 1200ORPM, and chops the VEVOR logo into fragments ("Vow") which the
# translator then dutifully renders as "Gelübde" over the brand. Every later
# stage is faithful to that input, so the fixes have to happen before the LLM:
#
# * brand regions are neither translated nor left erased -- the original pixels
#   are put back, because a translated logo is worse than an English one;
# * spec cells (62CC / 3.8HP / 12000RPM) skip the model entirely, so it cannot
#   "improve" a number it does not understand;
# * O/o and I/l inside a token that already contains digits are digit
#   confusions, not letters, and are repaired before anything sees them.

# A whole region that is just a value plus a short unit: 62CC, 3.8HP, +40%, 1h.
_LITERAL_SPEC = re.compile(r"^[+\-]?[\d.,]+\s*[A-Za-z%×x/]{0,4}$")

# Chart tick: a short number plus a one/two letter unit ("1h", "3 Std." no).
_TICK = re.compile(r"^\s*\d{1,3}\s*[A-Za-z]{1,2}\s*$")

# "Oh" is a misread "0h", not an interjection. Anchored to a unit letter so
# "Ours" and "Ohm" are untouched.
_OH_UNIT = re.compile(r"\b[Oo](?=[hms]\b)")

_NUMERIC_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.,%/*×]*")
_CONFUSABLE = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1"})

# A value plus the unit it is measured in: "62CC" -> ("62", "CC").
_SPEC_SPLIT = re.compile(r"^([+\-]?\d+(?:[.,]\d+)?)\s*([A-Za-z°]{0,6})$")

# Units that make a token a spec rather than a code. Only consulted to justify
# a digit repair, never to classify a token on its own.
_SPEC_UNITS = frozenset(
    """
    cc hp rpm kw w v a ah mah nm mm cm km m in ft kg g lb lbs bar psi l ml gal
    hz khz db btu mph kpa mpa pa ms h s min
    """.split()
)

# Letters EasyOCR swaps in for digits. Applied only when the token was not a
# spec before and becomes one with a known unit after, so "G2" stays "G2".
_DIGIT_LOOKALIKE = {"G": "6", "S": "5", "B": "8", "Z": "2", "T": "7", "A": "4", "Q": "0", "D": "0"}


def _norm_token(text):
    """Upper-case letters and digits only, for brand comparison."""
    return re.sub(r"[^0-9A-Za-z]", "", str(text or "")).upper()


def _lcs_length(a, b):
    """Longest common subsequence length."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for ch in a:
        current = [0]
        for col, other in enumerate(b, start=1):
            current.append(previous[col - 1] + 1 if ch == other else max(previous[col], current[col - 1]))
        previous = current
    return previous[-1]


def _subsequence_ratio(a, b):
    """2*LCS/(len(a)+len(b)): tolerates OCR dropping letters mid-word.

    ``SequenceMatcher`` compares contiguous blocks, so "VOW" vs "VEVOR" scores
    0.5 even though V-O-R is right there in order. An OCR-chopped logo loses
    letters *inside* the word, which is exactly what a subsequence keeps.
    """
    if not a or not b:
        return 0.0
    return 2.0 * _lcs_length(a, b) / (len(a) + len(b))


def is_brand_region(text, brands):
    """True when this region *is* a brand mark rather than copy about it.

    A sentence that merely mentions the brand ("Based on VEVOR Testing") must
    still be translated, so the loose matching is restricted to short tokens --
    that is exactly the shape an OCR-chopped logo arrives in.
    """
    token = _norm_token(text)
    if not token:
        return False
    for brand in brands:
        name = _norm_token(brand)
        if not name:
            continue
        if token == name:
            return True
        # "VEVOR®" / "VEVOR™" survive the punctuation strip as a longer token.
        if token.startswith(name) and len(token) <= len(name) + 2:
            return True
        if len(token) <= 5 and _subsequence_ratio(token, name) >= 0.6:
            return True
    return False


def is_brand_fragment(text, brands, box=None, max_height=0):
    """True when the region is a chopped brand mark rather than copy.

    EasyOCR cuts a logo into pieces and keeps the leading glyphs, so "VEVOR"
    arrives as "Vow" -- no longer a match, and far enough from the brand box
    that the overlap test cannot save it. What survives the cut is the first
    letter plus a few of the remaining ones in order, so that is the test here.

    ``max_height`` rejects body copy that happens to fit the pattern ("Very"):
    a logo caught inside the artwork is set smaller than the headline beside it,
    while running copy is set at body size.
    """
    token = _norm_token(text)
    if not 3 <= len(token) <= 5:
        return False
    if any(ch.isdigit() for ch in token):
        return False
    if box and max_height and (box[3] - box[1]) > max_height:
        return False
    for brand in brands:
        name = _norm_token(brand)
        if len(name) < 3 or token[0] != name[0]:
            continue
        if _lcs_length(token, name) >= 2 and _subsequence_ratio(token, name) >= 0.5:
            return True
    return False


def _spec_with_unit(text):
    """True when the token is a value plus a unit we know ("62CC", "3.8HP")."""
    match = _SPEC_SPLIT.match(str(text or "").strip())
    return bool(match) and match.group(2).lower() in _SPEC_UNITS


_LOST_DECIMAL_TWO_DIGIT = re.compile(r"^([1-9])(\d)(HP|KW|hp|kw|Hp|Kw)$")


def repair_lost_decimal_spec(text):
    """Restore a decimal point OCR dropped before a small power unit.

    EasyOCR often reads ``3.8HP`` as ``38HP``. Real two-digit engine ratings
    above ~20 HP are rare on consumer listing art; treat them as a missing dot
    between the first two digits instead.
    """
    token = str(text or "").strip()
    match = _LOST_DECIMAL_TWO_DIGIT.match(token)
    if not match:
        return token
    whole = int(match.group(1) + match.group(2))
    if whole <= 20:
        return token
    unit = match.group(3)
    if unit.isupper() or token.isupper():
        unit = unit.upper()
    elif unit.islower():
        unit = unit.lower()
    return f"{match.group(1)}.{match.group(2)}{unit}"


def repair_spec_digits(text):
    """Turn a letter that only makes sense as a digit back into a digit.

    "G2CC" is how EasyOCR renders "62CC" -- G and 6 share the same stroke. The
    swap is accepted only when the token was *not* a spec before and becomes
    one with a real unit after, so a genuine product code like "G2" is kept.
    """
    def fix(match):
        token = match.group(0)
        if not any(ch.isdigit() for ch in token) or _spec_with_unit(token):
            return token
        for position, ch in enumerate(token):
            digit = _DIGIT_LOOKALIKE.get(ch.upper())
            if not digit:
                continue
            trial = token[:position] + digit + token[position + 1:]
            if _spec_with_unit(trial):
                return trial
        return token

    return _NUMERIC_TOKEN.sub(fix, text)


def repair_numeric_ocr(text):
    """Fix letter/digit confusions inside tokens that are already numeric.

    Only tokens containing a digit are touched: "1200ORPM" -> "12000RPM", while
    "Ours" and "HP" are left alone. G/S/B are not mapped blindly -- "G2CC" may
    be a model code -- but they are when the fix reveals a known unit.
    """
    text = _OH_UNIT.sub("0", str(text or ""))

    def fix(match):
        token = match.group(0)
        if not any(ch.isdigit() for ch in token):
            return token
        token = token.translate(_CONFUSABLE)
        token = repair_lost_decimal_spec(token)
        return token

    return repair_spec_digits(_NUMERIC_TOKEN.sub(fix, text))


def is_noise_fragment(text):
    """True for OCR debris that carries no words: "@0", "|", stray punctuation."""
    stripped = str(text or "").strip()
    if not stripped:
        return True
    alnum = _norm_token(stripped)
    if not alnum:
        return True
    # A single alphanumeric character wrapped in symbols is a fragment of a
    # logo or a tick mark, never copy.
    return len(alnum) <= 1 and len(stripped) <= 3 and not stripped.isalnum()


def is_literal_spec(text):
    """True when the region is language-neutral (a number plus a unit)."""
    return bool(_LITERAL_SPEC.match(str(text or "").strip()))


def numeric_signature(text):
    """Digits only: 12.000 / 12,000 / 12000 all collapse to "12000"."""
    return re.sub(r"[^\d]", "", str(text or ""))


def numbers_preserved(source, translation):
    """True when every digit of the source survives in the translation.

    Separators are ignored on purpose: German writes 12.000 where the source has
    12000, which is correct CLDR, not a lost number. A digit that disappears
    entirely is a mangled spec and is worth falling back for.
    """
    want = numeric_signature(source)
    if not want:
        return True
    return want in numeric_signature(translation)


def _overlap_ratio(box_a, box_b):
    """Intersection over the smaller box: 1.0 when one box sits inside another."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    smaller = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
    return intersection / smaller if smaller > 0 else 0.0


def _contains(outer, inner, tolerance=6):
    """True when ``inner`` sits inside ``outer`` (with a small OCR tolerance)."""
    return (inner[0] >= outer[0] - tolerance and inner[1] >= outer[1] - tolerance
            and inner[2] <= outer[2] + tolerance and inner[3] <= outer[3] + tolerance)


def _box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _contained_duplicate(entry, pool):
    """Return the larger region that already covers this one, if any.

    EasyOCR often reports a headline and a slice of it as two regions ("40%*
    Faster Cutting" next to "+40%"). Rendering both paints the same words twice,
    which reads as ghosting. A duplicate is a *contained* box whose text is
    also part of the bigger region's text, so a tick label inside a chart never
    counts as one.
    """
    token = _norm_token(entry["text"])
    if not token:
        return None
    area = _box_area(entry["box"])
    for other in pool:
        if other is entry:
            continue
        other_token = _norm_token(other["text"])
        if not other_token or token not in other_token:
            continue
        if other_token == token and _box_area(other["box"]) <= area:
            continue
        if _contains(other["box"], entry["box"]):
            return other
    return None


def is_low_confidence_fragment(region, min_score=0.5, max_chars=4):
    """True for a very low-confidence short token: OCR noise, not copy.

    EasyOCR splits a stylised logo into pieces and reports them with poor
    confidence. Real short copy ("SALE", "NEW") comes back confident, so the
    threshold is what separates the two.
    """
    token = _norm_token(region["text"])
    if not token or len(token) > max_chars:
        return False
    if any(ch.isdigit() for ch in token):
        return False
    score = region.get("score") or 0.0
    return 0.0 < score < min_score


def classify_regions(regions, brands, repair_numbers=True, drop_noise=True,
                     drop_short_ticks=False, tick_max_width=180,
                     protect_logo_fragments=True, logo_max_height=64):
    """Split OCR regions into what to translate, render, restore and drop.

    Returns a dict of four lists plus the original index kept on every entry, so
    the prompt payload and the render payload can be different sizes while still
    agreeing on the keys the model answers with.
    """
    translate, render, protected, actions = [], [], [], []
    pending = []
    for position, region in enumerate(regions):
        index = region.get("index", position)
        source = region["text"]
        text = repair_numeric_ocr(source) if repair_numbers else source
        entry = {
            "index": index,
            "box": region["box"],
            "text": text,
            "score": region.get("score", 0.0),
        }
        if text != source:
            entry["source_raw"] = source
            actions.append({**entry, "action": "repair:number"})

        if is_brand_region(text, brands):
            protected.append(entry)
            actions.append({**entry, "action": "protect:brand"})
            continue

        if drop_noise and is_noise_fragment(text):
            actions.append({**entry, "action": "drop:noise"})
            continue

        pending.append(entry)

    # Second pass: a logo the OCR cut into pieces yields a fragment that shares
    # almost no characters with the brand name ("Vow" vs "VEVOR"), but it does
    # sit on top of the brand box. Geometry is the reliable signal here.
    brand_boxes = [entry["box"] for entry in protected]
    for entry in pending:
        overlapping = any(_overlap_ratio(entry["box"], box) >= 0.3 for box in brand_boxes)
        if overlapping:
            protected.append(entry)
            actions.append({**entry, "action": "protect:brand_overlap"})
            continue

        # The fragment can also land beside the brand box instead of on it. Then
        # only the text is left: same leading letter, rest still in order.
        if protect_logo_fragments and is_brand_fragment(
                entry["text"], brands, entry["box"], logo_max_height):
            protected.append(entry)
            actions.append({**entry, "action": "protect:brand_fragment"})
            continue

        owner = _contained_duplicate(entry, pending)
        if owner is not None:
            # Painting the headline and its slice separately prints it twice.
            actions.append({**entry, "action": "drop:overlap", "owner": owner["index"]})
            continue

        x1, _, x2, _ = entry["box"]
        if drop_short_ticks and _TICK.match(entry["text"]) and (x2 - x1) <= tick_max_width:
            actions.append({**entry, "action": "drop:tick"})
            continue

        if drop_noise and is_low_confidence_fragment(entry):
            actions.append({**entry, "action": "drop:low_confidence"})
            continue

        if is_literal_spec(entry["text"]):
            # Rendered verbatim, never sent to the translator.
            render.append({**entry, "literal": True})
            actions.append({**entry, "action": "literal:spec"})
            continue

        render.append(entry)
        translate.append(entry)

    return {
        "translate": translate,
        "render": render,
        "protected": protected,
        "actions": actions,
    }


def regions_to_payload(regions, width, height):
    """Serialise regions back into the labelme shape downstream nodes parse."""
    shapes = []
    for region in regions:
        shape = {
            "label": region["text"],
            "points": [[region["box"][0], region["box"][1]],
                       [region["box"][2], region["box"][3]]],
            "shape_type": "rectangle",
            "index": region.get("index"),
        }
        if region.get("literal"):
            # parse_ocr carries this through so the renderer copies the spec
            # instead of waiting for a translation that will never arrive.
            shape["literal"] = True
        shapes.append(shape)
    return json.dumps(
        {"imageWidth": width, "imageHeight": height, "shapes": shapes},
        ensure_ascii=False,
    )


def summarise_guard(result, brands):
    counts = {}
    for action in result["actions"]:
        counts[action["action"]] = counts.get(action["action"], 0) + 1
    lines = [
        f"translate={len(result['translate'])} render={len(result['render'])} "
        f"protected={len(result['protected'])} brands={','.join(brands)}"
    ]
    lines.append("  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for action in result["actions"][:40]:
        lines.append(
            f"  [{action['action']}] {action.get('source_raw', action['text'])!r} "
            f"-> {action['text']!r}"
        )
    if len(result["actions"]) > 40:
        lines.append(f"  ... {len(result['actions']) - 40} more")
    return "\n".join(lines)


class VevorOcrGuard:
    """Clean OCR regions before any model or renderer sees them."""

    DESCRIPTION = (
        "Guard the OCR hand-off: protect brand marks (never translated, pixels "
        "restored), copy language-neutral spec cells verbatim instead of asking a "
        "model, repair letter/digit OCR confusions inside numbers, and drop "
        "symbol-only debris. Emits separate translate/render/protected payloads "
        "that keep the original region indices."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ocr_json": ("JSON", {"forceInput": True}),
                "brands": ("STRING", {"default": "VEVOR,AUVII"}),
                "repair_numbers": ("BOOLEAN", {"default": True}),
                "drop_noise": ("BOOLEAN", {"default": True}),
                # Off by default: chart ticks are real content, and dropping them
                # silently loses the axis. Turn on only when the ticks are known
                # to be unreadable debris.
                "drop_short_ticks": ("BOOLEAN", {"default": False}),
                # A logo OCR chopped ("Vow" for VEVOR) is protected when it
                # keeps the brand's first letter and sits at logo size.
                "protect_logo_fragments": ("BOOLEAN", {"default": True}),
                "logo_max_height": ("INT", {"default": 64, "min": 0, "max": 512, "step": 2}),
            }
        }

    RETURN_TYPES = ("JSON", "JSON", "JSON", "STRING")
    RETURN_NAMES = ("prompt_json", "render_json", "protected_json", "summary")
    FUNCTION = "guard"
    CATEGORY = "vevor/text"

    def guard(self, ocr_json, brands, repair_numbers=True, drop_noise=True,
              drop_short_ticks=False, protect_logo_fragments=True, logo_max_height=64):
        regions, width, height = parse_ocr(ocr_json, with_index=True)
        names = [part.strip() for part in str(brands or "").split(",") if part.strip()]
        result = classify_regions(
            regions, names, repair_numbers=repair_numbers, drop_noise=drop_noise,
            drop_short_ticks=drop_short_ticks,
            protect_logo_fragments=protect_logo_fragments,
            logo_max_height=logo_max_height,
        )
        return (
            regions_to_payload(result["translate"], width, height),
            regions_to_payload(result["render"], width, height),
            regions_to_payload(result["protected"], width, height),
            summarise_guard(result, names),
        )


class VevorRestoreProtected:
    """Paste the original pixels back over protected regions of the clean plate.

    The erase model removes every glyph it is given, including the logo that OCR
    happened to catch. Compositing the untouched source rectangle back is
    deterministic: it does not depend on the mask batch order the way dropping
    mask frames would.
    """

    DESCRIPTION = (
        "Restore untouched source pixels over protected (brand) boxes on the "
        "cleared plate, so an erase pass cannot repaint or delete a logo."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "source": ("IMAGE",),
                "protected_json": ("JSON", {"forceInput": True}),
                "pad_px": ("INT", {"default": 6, "min": 0, "max": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "restore"
    CATEGORY = "vevor/text"

    def restore(self, image, source, protected_json, pad_px=6):
        if torch is None:  # pragma: no cover - ComfyUI always has torch
            raise RuntimeError("torch is required to restore regions")

        regions, canvas_w, canvas_h = parse_ocr(protected_json, with_index=True)
        frame = image[0] if image.dim() == 4 else image
        src_frame = source[0] if source.dim() == 4 else source

        plate = Image.fromarray(
            np.clip(255.0 * frame.cpu().numpy(), 0, 255).astype(np.uint8)
        ).convert("RGB")
        original = Image.fromarray(
            np.clip(255.0 * src_frame.cpu().numpy(), 0, 255).astype(np.uint8)
        ).convert("RGB")
        if original.size != plate.size:
            original = original.resize(plate.size, Image.LANCZOS)

        if not regions:
            out = np.asarray(plate).astype(np.float32) / 255.0
            return (torch.from_numpy(out).unsqueeze(0),)

        scale_x = plate.width / canvas_w if canvas_w else 1.0
        scale_y = plate.height / canvas_h if canvas_h else 1.0
        for region in regions:
            x1, y1, x2, y2 = region["box"]
            left = max(0, int(round(x1 * scale_x)) - pad_px)
            top = max(0, int(round(y1 * scale_y)) - pad_px)
            right = min(plate.width, int(round(x2 * scale_x)) + pad_px)
            bottom = min(plate.height, int(round(y2 * scale_y)) + pad_px)
            if right <= left or bottom <= top:
                continue
            plate.paste(original.crop((left, top, right, bottom)), (left, top))

        out = np.asarray(plate).astype(np.float32) / 255.0
        return (torch.from_numpy(out).unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "VevorTextRender": VevorTextRender,
    "VevorLetterboxToTarget": VevorLetterboxToTarget,
    "VevorPadSquare": VevorPadSquare,
    "VevorNameOutput": VevorNameOutput,
    "VevorTranslationPrompt": VevorTranslationPrompt,
    "VevorSaveNamed": VevorSaveNamed,
    "VevorOcrGuard": VevorOcrGuard,
    "VevorRestoreProtected": VevorRestoreProtected,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VevorTextRender": "Text Render (OCR boxes, fit + wrap)",
    "VevorLetterboxToTarget": "Letterbox to Target (margin mask)",
    "VevorPadSquare": "Pad to Square (no crop)",
    "VevorNameOutput": "Name Output (keyword-locale-slot)",
    "VevorTranslationPrompt": "Translation Prompt (editable, OCR → JSON map)",
    "VevorSaveNamed": "Save Named (connected prefix)",
    "VevorOcrGuard": "OCR Guard (brands / specs / noise)",
    "VevorRestoreProtected": "Restore Protected (brand pixels)",
}
