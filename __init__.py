"""Vevor mask utilities for ComfyUI.

Why this pack exists
--------------------
``Apply EasyOCR`` (ComfyUI-EasyOCR) returns its MASK output as a *batch with one
frame per detected text region* -- each frame is a full-size mask that is white
only inside that one box. Core ComfyUI has no node that folds a MASK batch down
to a single frame:

* ``MaskComposite`` raises "The expanded size of the tensor (1) must match the
  existing size (N)" when the two masks have different batch sizes, so it cannot
  be used to merge a batch.
* ``ImageCompositeMasked`` silently slices the mask batch down to the
  destination's batch size (``repeat_to_batch_size``), which keeps frame 0 and
  throws the rest away. A 22-region OCR pass therefore only composited the first
  region -- a silent, non-erroring wrong result.

``VevorMaskBatchUnion`` fills that gap: it collapses ``(B, H, W)`` to ``(1, H, W)``
by taking the per-pixel maximum, i.e. the union of every region.

Place it between the OCR MASK output and the grow/feather chain.
"""

import importlib.util
import os
import traceback

import torch

# ComfyUI loads a custom-node directory under a module name derived from its
# *path*, so neither `from .text_render import` nor `from text_render import`
# resolves here. Load the sibling file explicitly instead of relying on package
# semantics that only hold when the pack is imported as a real package.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_sibling(module_name, filename):
    path = os.path.join(_HERE, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_text_render():
    return _load_sibling("vevor_mask_union_text_render", "text_render.py")


try:
    _text_render = _load_text_render()
    _TEXT_CLASS_MAPPINGS = _text_render.NODE_CLASS_MAPPINGS
    _TEXT_DISPLAY_MAPPINGS = _text_render.NODE_DISPLAY_NAME_MAPPINGS
except Exception:  # noqa: BLE001 - surface the reason, keep mask union working
    # A failure here must not take the rest of the pack down with it; the
    # traceback is what makes the missing nodes diagnosable from the log.
    traceback.print_exc()
    _TEXT_CLASS_MAPPINGS = {}
    _TEXT_DISPLAY_MAPPINGS = {}

# The QA nodes depend on the same rendering helpers, so they are loaded
# independently: a broken eval module must not cost the rendering nodes.
try:
    _eval = _load_sibling("vevor_mask_union_eval", "eval.py")
    _EVAL_CLASS_MAPPINGS = _eval.NODE_CLASS_MAPPINGS
    _EVAL_DISPLAY_MAPPINGS = _eval.NODE_DISPLAY_NAME_MAPPINGS
except Exception:  # noqa: BLE001
    traceback.print_exc()
    _EVAL_CLASS_MAPPINGS = {}
    _EVAL_DISPLAY_MAPPINGS = {}

try:
    _ocr = _load_sibling("vevor_mask_union_ocr_providers", "ocr_providers.py")
    _OCR_CLASS_MAPPINGS = _ocr.NODE_CLASS_MAPPINGS
    _OCR_DISPLAY_MAPPINGS = _ocr.NODE_DISPLAY_NAME_MAPPINGS
except Exception:  # noqa: BLE001
    traceback.print_exc()
    _OCR_CLASS_MAPPINGS = {}
    _OCR_DISPLAY_MAPPINGS = {}

try:
    _generative = _load_sibling("vevor_mask_union_generative_pipeline", "generative_pipeline.py")
    _GENERATIVE_CLASS_MAPPINGS = _generative.NODE_CLASS_MAPPINGS
    _GENERATIVE_DISPLAY_MAPPINGS = _generative.NODE_DISPLAY_NAME_MAPPINGS
except Exception:  # noqa: BLE001
    traceback.print_exc()
    _GENERATIVE_CLASS_MAPPINGS = {}
    _GENERATIVE_DISPLAY_MAPPINGS = {}


class VevorMaskBatchUnion:
    """Fold a MASK batch (one frame per region) into a single union mask."""

    DESCRIPTION = (
        "Union a MASK batch into one frame (per-pixel max). Use after Apply EasyOCR, "
        "which emits one mask per detected text region, before GrowMask/FeatherMask."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",)}}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "union"
    CATEGORY = "vevor/mask"

    def union(self, mask):
        # A 2D mask (H, W) is already a single frame; just give it a batch dim.
        if mask.dim() == 2:
            return (mask.unsqueeze(0),)

        # Nothing detected: there is no size information left to rebuild a
        # frame from, so pass it through and let the caller see the empty batch.
        if mask.shape[0] == 0:
            return (mask,)

        if mask.shape[0] == 1:
            return (mask,)

        # amax over the batch dim == pixel-wise OR of every region mask.
        return (torch.amax(mask, dim=0, keepdim=True),)


NODE_CLASS_MAPPINGS = {
    "VevorMaskBatchUnion": VevorMaskBatchUnion,
    **_TEXT_CLASS_MAPPINGS,
    **_EVAL_CLASS_MAPPINGS,
    **_OCR_CLASS_MAPPINGS,
    **_GENERATIVE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VevorMaskBatchUnion": "Mask Batch Union (regions → 1)",
    **_TEXT_DISPLAY_MAPPINGS,
    **_EVAL_DISPLAY_MAPPINGS,
    **_OCR_DISPLAY_MAPPINGS,
    **_GENERATIVE_DISPLAY_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "VevorMaskBatchUnion"]
