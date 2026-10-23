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
  destination's batch size, which keeps frame 0 and throws the rest away. A
  22-region OCR pass therefore only composited the first region -- a silent,
  non-erroring wrong result.

``VevorMaskBatchUnion`` fills that gap: it collapses ``(B, H, W)`` to ``(1, H, W)``
by taking the per-pixel maximum, i.e. the union of every region.

Place it between the OCR MASK output and the grow/feather chain.
"""

import torch


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


NODE_CLASS_MAPPINGS = {"VevorMaskBatchUnion": VevorMaskBatchUnion}

NODE_DISPLAY_NAME_MAPPINGS = {"VevorMaskBatchUnion": "Mask Batch Union (regions → 1)"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "VevorMaskBatchUnion"]
