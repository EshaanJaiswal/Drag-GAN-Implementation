from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any, Optional

from flask import Flask, jsonify, request, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent
WEBAPP_ROOT = PROJECT_ROOT / "webapp"
FRONTEND_DIR = WEBAPP_ROOT / "frontend"
DEFAULT_OUTPUT_DIR = WEBAPP_ROOT / "outputs"
DEFAULT_UPLOAD_DIR = WEBAPP_ROOT / "uploads"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

RUNTIME_CONFIG = {
    "checkpoint": (PROJECT_ROOT / "ffhq.pt").resolve(),
    "output_dir": DEFAULT_OUTPUT_DIR.resolve(),
    "upload_dir": DEFAULT_UPLOAD_DIR.resolve(),
    "device": "cuda",
}

_ENGINE = None


def get_engine():
    global _ENGINE

    if _ENGINE is None:
        from webapp.backend.engine import DragEngine

        _ENGINE = DragEngine(
            checkpoint_path=Path(RUNTIME_CONFIG["checkpoint"]),
            output_dir=Path(RUNTIME_CONFIG["output_dir"]),
            upload_dir=Path(RUNTIME_CONFIG["upload_dir"]),
            device=str(RUNTIME_CONFIG["device"]),
        )

    return _ENGINE


def reset_engine() -> None:
    global _ENGINE
    _ENGINE = None


def parse_int(value: Any, field_name: str, min_value: Optional[int] = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc

    if min_value is not None and parsed < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}.")

    return parsed


def parse_float(value: Any, field_name: str, min_value: Optional[float] = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number.") from exc

    if min_value is not None and parsed < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}.")

    return parsed


def parse_optional_int(value: Any, field_name: str) -> Optional[int]:
    if value in (None, ""):
        return None
    return parse_int(value, field_name)


def resolve_checkpoint_path(checkpoint_arg: str) -> Path:
    raw_path = Path(checkpoint_arg).expanduser()

    if raw_path.is_absolute():
        return raw_path.resolve()

    cwd_candidate = Path.cwd() / raw_path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()

    project_candidate = PROJECT_ROOT / raw_path
    if project_candidate.exists():
        return project_candidate.resolve()

    return cwd_candidate.resolve()


def resolve_storage_path(path_arg: str) -> Path:
    raw_path = Path(path_arg).expanduser()

    if raw_path.is_absolute():
        return raw_path.resolve()

    return (Path.cwd() / raw_path).resolve()


def ok(data: Any):
    return jsonify({"ok": True, "data": data})


def fail(message: str, status_code: int = 400):
    return jsonify({"ok": False, "error": message}), status_code


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/outputs/<path:filename>")
def outputs(filename: str):
    return send_from_directory(Path(RUNTIME_CONFIG["output_dir"]), filename)


@app.route("/uploads/<path:filename>")
def uploads(filename: str):
    return send_from_directory(Path(RUNTIME_CONFIG["upload_dir"]), filename)


@app.route("/api/health", methods=["GET"])
def health_check():
    try:
        engine = get_engine()
        return ok({"status": "ready", "device": engine.device})
    except Exception as exc:
        traceback.print_exc()
        return fail(str(exc), status_code=500)


@app.route("/api/generate", methods=["POST"])
def generate_session():
    try:
        payload = request.get_json(silent=True) or {}
        seed = parse_optional_int(payload.get("seed"), "seed")
        generation_steps = parse_int(payload.get("generation_steps", 1), "generation_steps", min_value=1)

        engine = get_engine()
        result = engine.create_generated_session(seed=seed, generation_steps=generation_steps)
        return ok(result)
    except ValueError as exc:
        return fail(str(exc), status_code=400)
    except Exception as exc:
        traceback.print_exc()
        return fail(str(exc), status_code=500)


@app.route("/api/invert", methods=["POST"])
def invert_image():
    try:
        if "image" not in request.files:
            return fail("Missing image file in form-data under key 'image'.", status_code=400)

        image_file = request.files["image"]
        image_bytes = image_file.read()

        if len(image_bytes) == 0:
            return fail("Uploaded image is empty.", status_code=400)

        steps = parse_int(request.form.get("inversion_steps", 900), "inversion_steps", min_value=1)
        lr = parse_float(request.form.get("inversion_lr", 0.05), "inversion_lr", min_value=1e-12)
        noise_reg = parse_float(
            request.form.get("inversion_noise_reg", 3e4),
            "inversion_noise_reg",
            min_value=0.0,
        )
        fullres_weight = parse_float(
            request.form.get("inversion_fullres_weight", 1.0),
            "inversion_fullres_weight",
            min_value=1e-12,
        )
        lowres_weight = parse_float(
            request.form.get("inversion_lowres_weight", 0.25),
            "inversion_lowres_weight",
            min_value=0.0,
        )
        gradient_weight = parse_float(
            request.form.get("inversion_gradient_weight", 0.35),
            "inversion_gradient_weight",
            min_value=0.0,
        )

        engine = get_engine()
        result = engine.create_inversion_session(
            image_bytes=image_bytes,
            filename=image_file.filename or "upload.png",
            steps=steps,
            lr=lr,
            noise_regularize_weight=noise_reg,
            fullres_weight=fullres_weight,
            lowres_weight=lowres_weight,
            gradient_weight=gradient_weight,
        )
        return ok(result)
    except ValueError as exc:
        return fail(str(exc), status_code=400)
    except Exception as exc:
        traceback.print_exc()
        return fail(str(exc), status_code=500)


@app.route("/api/drag", methods=["POST"])
def run_drag():
    try:
        payload = request.get_json(silent=True) or {}

        session_id = payload.get("session_id")
        if not session_id:
            return fail("session_id is required.", status_code=400)

        pairs = payload.get("pairs")
        if not isinstance(pairs, list):
            return fail("pairs must be a list.", status_code=400)

        drag_steps = parse_int(payload.get("drag_steps", 50), "drag_steps", min_value=1)
        drag_lr = parse_float(payload.get("drag_lr", 0.002), "drag_lr", min_value=1e-12)

        engine = get_engine()
        result = engine.run_drag(
            session_id=str(session_id),
            pairs=pairs,
            drag_steps=drag_steps,
            drag_lr=drag_lr,
        )
        return ok(result)
    except ValueError as exc:
        return fail(str(exc), status_code=400)
    except Exception as exc:
        traceback.print_exc()
        return fail(str(exc), status_code=500)


@app.route("/api/session/<session_id>", methods=["GET"])
def session_summary(session_id: str):
    try:
        engine = get_engine()
        result = engine.get_session_summary(session_id)
        return ok(result)
    except ValueError as exc:
        return fail(str(exc), status_code=404)
    except Exception as exc:
        traceback.print_exc()
        return fail(str(exc), status_code=500)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local DragGAN webapp")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the web server")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind the web server")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ffhq.pt",
        help="Path to StyleGAN checkpoint file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to store generated images and feature artifacts",
    )
    parser.add_argument(
        "--upload-dir",
        type=str,
        default=str(DEFAULT_UPLOAD_DIR),
        help="Directory to store uploaded source images",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Preferred torch device",
    )
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    RUNTIME_CONFIG["checkpoint"] = resolve_checkpoint_path(args.checkpoint)
    RUNTIME_CONFIG["device"] = args.device
    RUNTIME_CONFIG["output_dir"] = resolve_storage_path(args.output_dir)
    RUNTIME_CONFIG["upload_dir"] = resolve_storage_path(args.upload_dir)

    Path(RUNTIME_CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(RUNTIME_CONFIG["upload_dir"]).mkdir(parents=True, exist_ok=True)

    reset_engine()
    get_engine()

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
