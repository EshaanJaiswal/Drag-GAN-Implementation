# DragGAN Implementation on StyleGAN2

An experimental implementation of the [DragGAN](https://arxiv.org/abs/2305.10894) point-based image manipulation algorithm, built on top of [rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch).

This was a semester research project exploring how latent-space optimization and intermediate feature maps can be used to interactively "drag" content in GAN-generated face images.

## Overview

**DragGAN** (Pan et al., 2023) enables interactive spatial manipulation of generated images. A user selects "handle" points and "target" points on an image; the algorithm then optimizes the generator's latent code so that the content at each handle point moves toward its target.

This project implements the core DragGAN algorithm — motion supervision and nearest-neighbor point tracking — on top of a pretrained StyleGAN2 generator using FFHQ weights. It does **not** implement the binary mask for locality control described in the paper, which is a known limitation.

## What I Implemented

This repository is a **fork** of [rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch). The StyleGAN2 architecture, training scripts, pretrained weights, and GAN inversion (`projector.py`) are from that project. My additions:

1. **Feature map extraction** (`model.py`): Modified the `Generator.forward()` method to optionally return the intermediate 256×256×128 feature map during synthesis.
2. **DragGAN optimization loop** (`drag_optimize.py`): The core algorithm — motion supervision loss, point tracking via cosine similarity, and W-space latent optimization.
3. **Point selection tools** (`point_picker.py`, integrated UI in `drag_optimize.py`): Matplotlib-based tools for interactively selecting handle/target points.

The following were built with AI assistance and are **not** my main technical contribution:

4. **Web demo** (`app.py`, `webapp/`): A Flask + vanilla JS interface for the full pipeline.
5. **Checkpoint converter** (`convert_ada_pkl_to_pt.py`): Utility for converting NVIDIA ADA-style `.pkl` files to the rosinality `.pt` format.
6. **CLI expansion** of `drag_optimize.py`: The argparse structure, projection caching, and image-mode support that grew the file from 105 to 961 lines.

**GAN inversion** (projecting real images into latent space) was already available in the base repository (`projector.py`). The version in `drag_optimize.py` adapts the same approach.

## Pipeline

```
Random Z ∈ ℝ⁵¹²                    OR    Real image
       │                                      │
       ▼                                      ▼
Mapping Network (8-layer MLP)          GAN Inversion (W+ optimization)
       │                                      │
       ▼                                      ▼
   W ∈ ℝ⁵¹²  ──────────────────────────>  w_opt (trainable)
                                              │
                                              ▼
                                    Frozen StyleGAN2 Generator
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                       1024×1024 image   256×256×128      Per-layer noise
                                        feature map       (fixed)
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                     User-selected      Handle feature    Target direction
                     point pairs        extraction        computation
                                              │
                                              ▼
                                    Motion Supervision Loss
                                    L1(f_shifted, f_original)
                                              │
                                              ▼
                                    Backprop → update w_opt
                                              │
                                              ▼
                                    Point Tracking
                                    (cosine similarity search)
                                              │
                                              ▼
                                    Repeat for 50 steps
                                              │
                                              ▼
                                    Final dragged image
```

## Core Method

### Motion Supervision

For each handle-target pair in each optimization step:

1. Extract the 128-dim feature vector at the handle position (detached — no gradient).
2. Compute a one-pixel step direction toward the target: `d = sign(target - handle)`.
3. Extract the feature vector at `handle + d` (with gradient).
4. Compute L1 loss between the two feature vectors.
5. Average losses across all pairs and backpropagate to update `w_opt`.

### Point Tracking

After each optimization step, the handle content may have shifted:

1. Normalize the original handle feature and all spatial features.
2. Compute cosine similarity between the handle feature and every position in the 256×256 map.
3. The position with maximum similarity becomes the new handle location.

### What Is Optimized

- **Optimized**: `w_opt` — the W-space (or W+) latent code, shape `[1, 512]` or `[1, 18, 512]`.
- **Frozen**: All generator weights, per-layer noise buffers.
- **Optimizer**: Adam, lr=0.002.
- **Stopping**: Fixed 50 iterations (no convergence criterion).

## Repository Structure

```
├── model.py                  # StyleGAN2 Generator/Discriminator (modified for feature extraction)
├── drag_optimize.py          # Core DragGAN: CLI interface + optimization loop
├── point_picker.py           # Standalone matplotlib point picker
├── test_script.py            # Smoke test for feature map extraction
├── convert_ada_pkl_to_pt.py  # ADA .pkl → rosinality .pt converter
├── app.py                    # Flask web server entry point
├── webapp/
│   ├── backend/engine.py     # DragEngine class (web-facing wrapper)
│   ├── frontend/             # HTML + CSS + JS web UI
│   └── requirements.txt      # Web-specific dependencies
├── Initial_Drag_Examples/    # Before/after example outputs
│
│  ── Existing rosinality/stylegan2-pytorch files ──
├── train.py                  # StyleGAN2 training (not used in this project)
├── generate.py               # Sample generation
├── projector.py              # Original GAN inversion (LPIPS-based)
├── op/                       # Custom CUDA kernels (upfirdn2d, fused_act)
├── lpips/                    # Perceptual loss library
└── ...                       # Other rosinality utilities
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (tested with CUDA 10.1/10.2)
- PyTorch 1.3.1+ with CUDA support

### Setup

```bash
git clone https://github.com/EshaanJaiswal/Drag-GAN-Implementation.git
cd Drag-GAN-Implementation

pip install torch torchvision  # Install PyTorch for your CUDA version
pip install Pillow matplotlib  # For image handling and point picker
```

### Pretrained Weights

Download the FFHQ pretrained checkpoint and place it in the repository root as `ffhq.pt`.

The rosinality pretrained checkpoints are available at: [Google Drive link](https://drive.google.com/open?id=1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO)

## Usage

### CLI — Generate and Drag

```bash
# Generate a face, pick points via matplotlib UI, run 50 drag steps
python drag_optimize.py

# With specific handle/target coordinates (in 256×256 feature-map space)
python drag_optimize.py --handle 120 150 --target 120 170

# Multiple point pairs
python drag_optimize.py --handle 120 150 --handle 80 100 --target 120 170 --target 80 130

# With a fixed seed for reproducibility
python drag_optimize.py --seed 42 --handle 120 150 --target 120 170
```

### CLI — Drag on an Existing Image

```bash
# Auto-project an existing image, then drag
python drag_optimize.py --image-mode existing --existing-image photo.png --handle 120 150 --target 120 170

# Control projection quality
python drag_optimize.py --image-mode existing --existing-image photo.png --project-steps 900
```

### Point Picker (Standalone)

```bash
# Generate an image first, then pick points interactively
python point_picker.py --image drag_step_00.png --num-points 2
```

### Web Demo

```bash
pip install flask
python app.py --host 0.0.0.0 --port 7860 --checkpoint ffhq.pt
# Open http://localhost:7860
```

## Results / Observations

### Identity Bleed

Without a binary mask constraining the optimization region, large drags cause the overall face identity to change. The latent code `w_opt` is global — it controls the entire image — so moving one feature can inadvertently alter others.

Example outputs are in the `Initial_Drag_Examples/` directory, showing before/after comparisons at different drag distances.

### Where the Mask Would Help

The DragGAN paper applies a binary mask M so that only features within the masked region contribute to the loss:

```
L_masked = L1(M ⊙ f_shifted, M ⊙ f_original)
```

This constrains gradients to the region of interest, preserving identity in unmasked areas. This is **not implemented** in this project.

## Limitations

1. **No binary mask**: The main source of identity bleed. All spatial positions affect the loss equally.
2. **Fixed iteration count**: 50 steps with no convergence-based stopping.
3. **Global feature search**: Point tracking searches the entire 256×256 map instead of a local neighborhood, which can cause tracking jumps.
4. **FFHQ-only**: The pretrained model only generates faces. The algorithm is domain-agnostic, but a different generator and weights would be needed for other domains.
5. **No quantitative evaluation**: Results are assessed qualitatively; no FID, identity similarity, or other metrics are computed.

## References

- **DragGAN**: Xingang Pan, Ayush Tewari, Thomas Leimkühler, Lingjie Liu, Abhimitra Meka, Christian Theobalt. [Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold](https://arxiv.org/abs/2305.10894). SIGGRAPH 2023.
- **StyleGAN2**: Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila. [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958). CVPR 2020.

## Attribution

- **StyleGAN2 PyTorch implementation**: [rosinality/stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch) by Kim Seonghyeon (MIT License).
- **Custom CUDA kernels**: From the [official NVIDIA StyleGAN2 repository](https://github.com/NVlabs/stylegan2) (NVIDIA Source Code License).
- **LPIPS**: From [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) (BSD 2-Clause License).
- **FID Inception V3**: From [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid).

## License

The original rosinality code is MIT Licensed (see [LICENSE](LICENSE)). Additional license files for NVIDIA code and LPIPS are included in the repository.
