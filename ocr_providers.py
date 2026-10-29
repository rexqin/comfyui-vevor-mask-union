"""Optional OCR backends that emit the same labelme JSON + MASK batch as Apply EasyOCR."""

from __future__ import annotations

import json
import re

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


def _require_torch():
    if torch is None:
        raise RuntimeError("torch is required for Vevor OCR nodes")


def _frame_rgb_uint8(image):
    _require_torch()
    if image.dim() != 4 or image.shape[0] < 1:
        raise ValueError("OCR expects IMAGE batch with at least one frame")
    frame = image[0].detach().cpu().numpy()
    return np.clip(frame * 255.0, 0, 255).astype(np.uint8)


def _box_from_quad(quad):
    xs = [float(p[0]) for p in quad]
    ys = [float(p[1]) for p in quad]
    return min(xs), min(ys), max(xs), max(ys)


def _labelme_doc(width, height, entries):
    """entries: list of (text, score, x1, y1, x2, y2)."""
    shapes = []
    for idx, (text, score, x1, y1, x2, y2) in enumerate(entries):
        label = f"{text}:{score:.4f}" if score is not None else str(text)
        shapes.append(
            {
                "label": label,
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
                "index": idx,
            }
        )
    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": "",
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


def _mask_batch_from_entries(entries, height, width):
    _require_torch()
    if not entries:
        return torch.zeros((0, height, width), dtype=torch.float32)
    planes = []
    for _text, _score, x1, y1, x2, y2 in entries:
        plane = np.zeros((height, width), dtype=np.float32)
        left = max(0, int(round(x1)))
        top = max(0, int(round(y1)))
        right = min(width, int(round(x2)))
        bottom = min(height, int(round(y2)))
        if right > left and bottom > top:
            plane[top:bottom, left:right] = 1.0
        planes.append(plane)
    stacked = np.stack(planes, axis=0)
    return torch.from_numpy(stacked)


def _pack_ocr_output(entries, width, height):
    doc = _labelme_doc(width, height, entries)
    mask = _mask_batch_from_entries(entries, height, width)
    return [doc], mask


def _bbox_from_polygon_or_bbox(bbox, polygon):
    if bbox and len(bbox) >= 4:
        return [float(v) for v in bbox[:4]]
    if polygon and len(polygon) >= 3:
        xs = [float(p[0]) for p in polygon]
        ys = [float(p[1]) for p in polygon]
        return [min(xs), min(ys), max(xs), max(ys)]
    return None


def _parse_surya_full_json(text):
    """Parse PDF-Tools ``SuryaOCRLayout`` ``full_data_json`` into OCR entries."""
    raw = str(text or "").strip()
    if not raw:
        return []
    data = json.loads(raw)
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(data["error"])
    entries = []
    for page in data.get("ocr") or []:
        for line in page.get("text_lines") or []:
            text_val = str(line.get("text") or "").strip()
            if not text_val:
                continue
            box = _bbox_from_polygon_or_bbox(line.get("bbox"), line.get("polygon"))
            if not box:
                continue
            x1, y1, x2, y2 = box
            entries.append((text_val, 0.9, x1, y1, x2, y2))
    return entries


def _parse_vlm_json(text):
    """Extract a JSON array of {text, bbox:[x1,y1,x2,y2]} from model output."""
    raw = str(text or "").strip()
    if not raw:
        return []
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if fence:
        raw = fence.group(1).strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    data = json.loads(raw)
    entries = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text_val = str(item.get("text") or item.get("label") or "").strip()
        bbox = item.get("bbox") or item.get("box") or item.get("points")
        if not text_val or not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        score = float(item.get("score", 0.9))
        entries.append((text_val, score, x1, y1, x2, y2))
    return entries


_PADDLE_ENGINE = None
_SURYA_BUNDLE = None

VL_OCR_JSON_PROMPT = (
    "List every visible text region in this advertisement image as a JSON array. "
    'Each item: {"text": string, "bbox": [x1,y1,x2,y2], "score": float}. '
    "Use pixel coordinates on this image. Include titles, specs, chart labels, "
    "footnotes. Do not merge regions. Output JSON only."
)


class VevorOcrPaddle:
    DESCRIPTION = (
        "Run PaddleOCR locally; output labelme JSON + per-region MASK batch "
        "(same contract as Apply EasyOCR). Requires: pip install paddlepaddle-gpu paddleocr"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lang": (["en", "ch", "chinese_cht", "japan", "korean"], {"default": "en"}),
                "use_gpu": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("JSON", "MASK")
    RETURN_NAMES = ("ocr_json", "mask")
    FUNCTION = "run"
    CATEGORY = "vevor/ocr"

    def run(self, image, lang, use_gpu):
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR not installed in ComfyUI Python. "
                "On the server: pip install paddlepaddle-gpu paddleocr"
            ) from exc

        global _PADDLE_ENGINE
        rgb = _frame_rgb_uint8(image)
        height, width = rgb.shape[0], rgb.shape[1]
        if _PADDLE_ENGINE is None:
            kwargs = {"use_angle_cls": True, "lang": lang}
            try:
                _PADDLE_ENGINE = PaddleOCR(**kwargs, use_gpu=use_gpu, show_log=False)
            except TypeError:
                device = "gpu:0" if use_gpu else "cpu"
                _PADDLE_ENGINE = PaddleOCR(**kwargs, device=device)
        ocr = _PADDLE_ENGINE
        result = ocr.ocr(rgb, cls=True)
        entries = []
        lines = result[0] if result else []
        for line in lines or []:
            if not line or len(line) < 2:
                continue
            quad, rec = line[0], line[1]
            text = rec[0] if isinstance(rec, (list, tuple)) else str(rec)
            score = float(rec[1]) if isinstance(rec, (list, tuple)) and len(rec) > 1 else 0.9
            x1, y1, x2, y2 = _box_from_quad(quad)
            entries.append((str(text).strip(), score, x1, y1, x2, y2))
        payload, mask = _pack_ocr_output(entries, width, height)
        return (payload, mask)


class VevorOcrSurya:
    DESCRIPTION = (
        "Run Surya OCR locally; labelme JSON + MASK batch. "
        "Requires: pip install surya-ocr"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lang": (["en", "de", "fr", "es", "it", "zh", "ja"], {"default": "en"}),
            }
        }

    RETURN_TYPES = ("JSON", "MASK")
    RETURN_NAMES = ("ocr_json", "mask")
    FUNCTION = "run"
    CATEGORY = "vevor/ocr"

    def run(self, image, lang):
        try:
            from surya.detection import batch_text_detection
            from surya.model.detection.model import load_model as load_det_model
            from surya.model.detection.processor import load_processor as load_det_processor
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor
            from surya.recognition import batch_recognition
        except ImportError as exc:
            raise RuntimeError(
                "Surya not installed. On the ComfyUI server: pip install surya-ocr"
            ) from exc

        if Image is None:
            raise RuntimeError("PIL is required for Surya OCR")

        rgb = _frame_rgb_uint8(image)
        height, width = rgb.shape[0], rgb.shape[1]
        pil = Image.fromarray(rgb)

        global _SURYA_BUNDLE
        if _SURYA_BUNDLE is None:
            _SURYA_BUNDLE = (
                load_det_model(),
                load_det_processor(),
                load_rec_model(),
                load_rec_processor(),
            )
        det_model, det_processor, rec_model, rec_processor = _SURYA_BUNDLE

        det = batch_text_detection([pil], det_model, det_processor)[0]
        rec = batch_recognition([pil], [det], rec_model, rec_processor, langs=[[lang]])[0]

        entries = []
        for line in rec.text_lines:
            text = str(line.text or "").strip()
            if not text:
                continue
            x1, y1, x2, y2 = line.bbox
            score = float(getattr(line, "confidence", 0.9) or 0.9)
            entries.append((text, score, x1, y1, x2, y2))
        payload, mask = _pack_ocr_output(entries, width, height)
        return (payload, mask)


class VevorSuryaJsonToOcrJson:
    """Parse PDF-Tools ``SuryaOCRLayout`` ``full_data_json`` for VevorOcrGuard."""

    DESCRIPTION = (
        "Wire after community SuryaOCRLayout (PDF-Tools): full_data_json → labelme JSON + MASK."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "full_data_json": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("JSON", "MASK")
    RETURN_NAMES = ("ocr_json", "mask")
    FUNCTION = "convert"
    CATEGORY = "vevor/ocr"

    def convert(self, image, full_data_json):
        rgb = _frame_rgb_uint8(image)
        height, width = rgb.shape[0], rgb.shape[1]
        entries = _parse_surya_full_json(full_data_json)
        payload, mask = _pack_ocr_output(entries, width, height)
        return (payload, mask)


class VevorVlTextToOcrJson:
    """Parse ComfyUI-Qwen-VL ``QwenVisionParser`` text into labelme JSON + MASK batch."""

    DESCRIPTION = (
        "Wire after community QwenVisionParser: converts VL JSON-in-text to the same "
        "labelme payload Apply EasyOCR uses (for VevorOcrGuard)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("JSON", "MASK")
    RETURN_NAMES = ("ocr_json", "mask")
    FUNCTION = "convert"
    CATEGORY = "vevor/ocr"

    def convert(self, image, text):
        rgb = _frame_rgb_uint8(image)
        height, width = rgb.shape[0], rgb.shape[1]
        entries = _parse_vlm_json(text)
        payload, mask = _pack_ocr_output(entries, width, height)
        return (payload, mask)


NODE_CLASS_MAPPINGS = {
    "VevorOcrPaddle": VevorOcrPaddle,
    "VevorSuryaJsonToOcrJson": VevorSuryaJsonToOcrJson,
    "VevorVlTextToOcrJson": VevorVlTextToOcrJson,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VevorOcrPaddle": "OCR · PaddleOCR → JSON+mask (no registry JSON node yet)",
    "VevorSuryaJsonToOcrJson": "Surya JSON → OCR JSON (after PDF-Tools)",
    "VevorVlTextToOcrJson": "VL text → OCR JSON (after Qwen-VL)",
}
