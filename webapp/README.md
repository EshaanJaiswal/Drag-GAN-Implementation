# DragGAN Local Webapp

This folder contains a standalone web application for local hosting on a GPU server.
It does not modify the original project scripts.

## Features

- Generate an image and create a drag session.
- Upload an image and run GAN inversion to create a drag session.
- Select multiple handle-target point pairs directly in the browser.
- Run drag optimization with configurable drag steps and learning rate.
- Save and preview feature maps for both initial (input/generation) and output states.
- Download raw feature tensors (`.pt`) for initial and output states.

## Folder Layout

- `../app.py`: Primary Flask server entrypoint (run from project root).
- `backend/app.py`: Backward-compatible forwarding entrypoint.
- `backend/engine.py`: Model loading, generation, inversion, drag logic, and artifact saving.
- `frontend/index.html`: Main web UI.
- `frontend/app.js`: Client-side interactions and API calls.
- `frontend/styles.css`: Responsive UI styles.
- `outputs/`: Generated images, feature maps, tensors, and latent bundles.
- `uploads/`: Uploaded source images for inversion sessions.

## Requirements

Install dependencies in your active environment:

```bash
pip install -r webapp/requirements.txt
```

## Run

From the repository root:

```bash
python app.py --host 0.0.0.0 --port 7860 --checkpoint ffhq.pt --device cuda
```

Optional storage locations can be configured without hardcoded paths:

```bash
python app.py --output-dir webapp/outputs --upload-dir webapp/uploads
```

Then open:

- `http://localhost:7860` on the same machine.
- `http://<server-ip>:7860` from another machine if firewall/network allow it.

## API Summary

- `POST /api/generate`
  - JSON body:
    - `seed` (optional int)
    - `generation_steps` (int >= 1)

- `POST /api/invert`
  - `multipart/form-data`:
    - `image` (required file)
    - `inversion_steps` (int >= 1)
    - `inversion_lr` (float > 0)
    - `inversion_noise_reg` (float >= 0)
    - `inversion_fullres_weight` (float > 0)
    - `inversion_lowres_weight` (float >= 0)
    - `inversion_gradient_weight` (float >= 0)

- `POST /api/drag`
  - JSON body:
    - `session_id` (string)
    - `pairs` (array of `{ handle: {x, y}, target: {x, y} }`)
    - `drag_steps` (int >= 1)
    - `drag_lr` (float > 0)

## Notes for GPU Servers

- The app loads the generator once and keeps it in memory.
- Requests are processed one at a time for GPU safety.
- If CUDA is unavailable, it falls back to CPU when `--device cuda` is requested.
