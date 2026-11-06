"""ApiYi image edits (design-agent ``callApiYiImageEdit`` equivalent)."""

from __future__ import annotations

import io
import json
import os
import sys
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

APIYI_EDITS_URL = "https://api.apiyi.com/v1/images/edits"
LONG_EDGE_MIN = 1024
LONG_EDGE_MAX = 1536


def nearest16(n: float) -> int:
    return max(16, int(round(n / 16) * 16))


def api_yi_size_for(width: int, height: int) -> tuple[int, int]:
    long_edge = max(width, height, 1)
    scale = min(LONG_EDGE_MAX / long_edge, max(1.0, LONG_EDGE_MIN / long_edge))
    return nearest16(width * scale), nearest16(height * scale)


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    arr = image.cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    rgb = np.clip(arr[..., :3] * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    rgb = img.convert("RGB")
    arr = np.array(rgb).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _multipart_form(fields: list[tuple[str, str]], files: list[tuple[str, str, bytes, str]]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    lines: list[bytes] = []
    for name, value in fields:
        lines.append(f"--{boundary}\r\n".encode())
        lines.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        lines.append(f"{value}\r\n".encode())
    for name, filename, data, mime in files:
        lines.append(f"--{boundary}\r\n".encode())
        lines.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        )
        lines.append(f"Content-Type: {mime}\r\n\r\n".encode())
        lines.append(data)
        lines.append(b"\r\n")
    lines.append(f"--{boundary}--\r\n".encode())
    body = b"".join(lines)
    return body, boundary


def _fetch_url_bytes(url: str, timeout: float) -> bytes:
    req = Request(url, headers={"User-Agent": "vevor-mask-union/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _call_apiyi_edits(
    *,
    api_key: str,
    model: str,
    prompt: str,
    canvas_png: bytes,
    target_w: int,
    target_h: int,
    quality: str,
    timeout: float = 600.0,
) -> bytes:
    fields = [
        ("model", model),
        ("prompt", prompt),
        ("size", f"{target_w}x{target_h}"),
        ("quality", quality),
        ("output_format", "png"),
    ]
    body, boundary = _multipart_form(
        fields,
        [("image", "original.png", canvas_png, "image/png")],
    )
    req = Request(
        APIYI_EDITS_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:200]
        raise RuntimeError(f"apiyi 生图失败（HTTP {e.code}）：{err_body}") from e
    except URLError as e:
        raise RuntimeError(f"apiyi 请求失败：{e}") from e

    out: bytes | None = None
    try:
        payload: Any = json.loads(text)
        item = (payload.get("data") or [None])[0] if isinstance(payload, dict) else None
        if isinstance(item, dict):
            if item.get("b64_json"):
                import base64

                out = base64.b64decode(str(item["b64_json"]))
            elif item.get("url"):
                out = _fetch_url_bytes(str(item["url"]), 120.0)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        raise RuntimeError(f"apiyi 返回无法解析：{text[:200]}") from e
    if not out:
        raise RuntimeError("apiyi 返回为空（既没有 b64_json 也没有 url）")
    return out


class VevorApiYiImageEdit:
    """Single-image generative edit via ApiYi (GPT Image 2.5 Sunburst)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"forceInput": True}),
                "api_key": ("STRING", {"default": "REPLACE_APIYI_API_KEY"}),
                "model": ("STRING", {"default": "gpt-image-2.5-sunburst"}),
                "quality": (["low", "medium", "high"], {"default": "high"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "vevor/generative"

    def edit(self, image, prompt: str, api_key: str, model: str, quality: str = "high"):
        """HTTP edits only — input IMAGE should already be canvas-sized (e.g. 1hew resize)."""
        key = str(api_key or "").strip()
        if not key or key.startswith("REPLACE_"):
            env = os.environ.get("APIYI_API_KEY", "").strip()
            if not env:
                raise RuntimeError("APIYI API key 未配置（节点 api_key 或环境变量 APIYI_API_KEY）")
            key = env

        pil = _tensor_to_pil(image)
        tw, th = pil.size
        if tw < 1 or th < 1:
            raise RuntimeError("输入图像尺寸无效")

        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        canvas_png = buf.getvalue()

        out_png = _call_apiyi_edits(
            api_key=key,
            model=str(model or "gpt-image-2.5-sunburst").strip(),
            prompt=str(prompt or ""),
            canvas_png=canvas_png,
            target_w=tw,
            target_h=th,
            quality=quality or "high",
        )

        out_img = Image.open(io.BytesIO(out_png)).convert("RGB")
        return (_pil_to_tensor(out_img),)


NODE_CLASS_MAPPINGS = {
    "VevorApiYiImageEdit": VevorApiYiImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VevorApiYiImageEdit": "Generative · ApiYi Image Edit",
}
