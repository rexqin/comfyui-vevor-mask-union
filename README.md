# comfyui-vevor-mask-union

`VevorMaskBatchUnion` — fold a MASK **batch** into a single **union** mask.

## Why

`Apply EasyOCR` (ComfyUI-EasyOCR) emits one mask frame per detected text region:
a batch of `(N, H, W)`, each frame white only inside its own region.

Core ComfyUI cannot reduce that batch:

- `MaskComposite` raises `The expanded size of the tensor (1) must match the existing size (N)`
  when the two masks differ in batch size, so it cannot merge a batch.
- `ImageCompositeMasked` silently slices the mask batch down to the destination's
  batch size. With a single destination image it keeps **frame 0 only** and drops
  the rest — so a 22-region OCR pass composites just 1 region, with no error.

This node fills the gap with `torch.amax(mask, dim=0)`.

## Wiring

```
Apply EasyOCR (MASK) → VevorMaskBatchUnion → GrowMask → FeatherMask → ImageCompositeMasked
```

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/rexqin/comfyui-vevor-mask-union
# restart ComfyUI
```

Or via ComfyUI-Manager → *Install via Git URL*.

## Notes

- `(B, H, W)` → `(1, H, W)`; a 2D `(H, W)` input is passed through with a batch dim added.
- An empty batch is returned unchanged: there is no size information left to rebuild a frame from.
