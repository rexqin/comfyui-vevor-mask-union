"""Decode OpenRouter / partner LLM STRING responses into Comfy IMAGE tensors."""

from __future__ import annotations

import base64
import io
import json
import re
from urllib.request import Request, urlopen

import numpy as np
import torch
from PIL import Image


def _fetch_url_bytes(url: str, timeout: float) -> bytes:
    req = Request(url, headers={"User-Agent": "vevor-mask-union/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    rgb = img.convert("RGB")
    arr = np.array(rgb).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _decode_image_bytes(raw: str) -> bytes | None:
    text = str(raw or "").strip()
    if not text:
        return None

    if text.startswith("data:image"):
        m = re.match(r"^data:image/[^;]+;base64,(.+)$", text, re.I | re.S)
        if m:
            try:
                return base64.b64decode(m.group(1).strip())
            except (ValueError, TypeError):
                return None

    if text.startswith("{") or text.startswith("["):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            data = obj.get("data")
            if isinstance(data, list) and data:
                item = data[0]
                if isinstance(item, dict):
                    if item.get("b64_json"):
                        try:
                            return base64.b64decode(str(item["b64_json"]))
                        except (ValueError, TypeError):
                            return None
                    url = item.get("url")
                    if url:
                        return _fetch_url_bytes(str(url), 120.0)

    if re.match(r"^https?://", text, re.I):
        return _fetch_url_bytes(text.split()[0], 120.0)

    if len(text) > 256 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", text):
        try:
            return base64.b64decode(re.sub(r"\s+", "", text))
        except (ValueError, TypeError):
            return None

    return None


class VevorGenerativeImageFromLlmResponse:
    """Turn OpenRouter LLM STRING (data URL / JSON / URL / base64) into IMAGE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_response": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "resolve"
    CATEGORY = "vevor/generative"

    def resolve(self, llm_response: str):
        decoded = _decode_image_bytes(llm_response)
        if not decoded:
            preview = str(llm_response or "").strip()
            if len(preview) > 240:
                preview = preview[:240] + "…"
            raise RuntimeError(
                "OpenRouter/LLM 响应未含可解码图片（需 data URL、JSON data[].b64_json|url、"
                f"http(s) URL 或足够长的 base64）。响应开头: {preview!r}"
            )
        try:
            out_img = Image.open(io.BytesIO(decoded)).convert("RGB")
        except OSError as exc:
            raise RuntimeError("LLM 响应解码为字节但无法作为图片打开") from exc
        return (_pil_to_tensor(out_img),)


NODE_CLASS_MAPPINGS = {
    "VevorGenerativeImageFromLlmResponse": VevorGenerativeImageFromLlmResponse,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VevorGenerativeImageFromLlmResponse": "Generative · Image from LLM response",
}
