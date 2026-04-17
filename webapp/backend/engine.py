from __future__ import annotations

import os
import sys
import threading
import time
import uuid
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from torch import optim
from torch.nn import functional as F
from torchvision import transforms, utils

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import Generator


def save_image_compat(tensor: torch.Tensor, path: Path) -> None:
    try:
        utils.save_image(tensor, str(path), normalize=True, value_range=(-1, 1))
    except TypeError:
        utils.save_image(tensor, str(path), normalize=True, range=(-1, 1))


def clamp_point(y: int, x: int, feature_h: int, feature_w: int) -> Tuple[int, int]:
    clamped_y = max(0, min(int(y), feature_h - 1))
    clamped_x = max(0, min(int(x), feature_w - 1))
    return clamped_y, clamped_x


def get_lr(t: float, initial_lr: float, rampdown: float = 0.25, rampup: float = 0.05) -> float:
    lr_ramp = min(1.0, (1.0 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1.0, t / rampup)
    return initial_lr * lr_ramp


def latent_noise(latent: torch.Tensor, strength: float) -> torch.Tensor:
    noise = torch.randn_like(latent) * strength
    return latent + noise


def image_gradients(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]
    grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]
    return grad_x, grad_y


def noise_regularize(noises: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(noises) == 0:
        return torch.tensor(0.0)

    loss = torch.tensor(0.0, device=noises[0].device)

    for noise in noises:
        size = noise.shape[2]
        current_noise = noise

        while True:
            loss = loss + (current_noise * torch.roll(current_noise, shifts=1, dims=3)).mean().pow(2)
            loss = loss + (current_noise * torch.roll(current_noise, shifts=1, dims=2)).mean().pow(2)

            if size <= 8:
                break

            current_noise = current_noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            current_noise = current_noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises: Sequence[torch.Tensor]) -> None:
    for noise in noises:
        mean = noise.mean()
        std = noise.std()
        noise.data.add_(-mean).div_(std + 1e-8)


@dataclass
class DragSession:
    session_id: str
    w: torch.Tensor
    noises: Optional[List[torch.Tensor]]
    image_height: int
    image_width: int
    feature_height: int
    feature_width: int
    mode: str
    seed: Optional[int]
    generation_steps: Optional[int]
    source_image_url: Optional[str]
    initial_image_url: str
    initial_feature_map_url: str
    initial_feature_tensor_url: str
    last_output_image_url: Optional[str] = None
    last_output_feature_map_url: Optional[str] = None
    last_output_feature_tensor_url: Optional[str] = None


class DragEngine:
    def __init__(
        self,
        checkpoint_path: Path,
        output_dir: Path,
        upload_dir: Path,
        device: str = "cuda",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.output_dir = Path(output_dir)
        self.upload_dir = Path(upload_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        self.device = self._resolve_device(device)
        self.g_ema = self._load_generator()

        self._sessions: Dict[str, DragSession] = {}
        self._sessions_lock = threading.Lock()
        self._gpu_lock = threading.Lock()

    def _resolve_device(self, requested: str) -> str:
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        if requested == "cpu":
            return "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return requested

    def _load_generator(self) -> Generator:
        g_ema = Generator(1024, 512, 8).to(self.device)
        checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device)

        if "g_ema" not in checkpoint:
            raise KeyError(f"Key 'g_ema' not found in checkpoint: {self.checkpoint_path}")

        g_ema.load_state_dict(checkpoint["g_ema"])
        g_ema.eval()
        return g_ema

    def _new_session_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def _next_path(self, base_dir: Path, session_id: str, label: str, suffix: str) -> Path:
        stamp = int(time.time() * 1000)
        token = uuid.uuid4().hex[:6]
        return base_dir / f"{session_id}_{label}_{stamp}_{token}{suffix}"

    def _to_output_url(self, path: Path) -> str:
        return f"/outputs/{path.name}"

    def _to_upload_url(self, path: Path) -> str:
        return f"/uploads/{path.name}"

    def _save_image(self, image_tensor: torch.Tensor, session_id: str, label: str) -> Tuple[Path, str]:
        path = self._next_path(self.output_dir, session_id, label, ".png")
        save_image_compat(image_tensor, path)
        return path, self._to_output_url(path)

    def _save_feature_map_outputs(
        self,
        feature_tensor: torch.Tensor,
        session_id: str,
        label: str,
    ) -> Tuple[Path, str, Path, str]:
        tensor_path = self._next_path(self.output_dir, session_id, f"{label}_features", ".pt")
        preview_path = self._next_path(self.output_dir, session_id, f"{label}_featuremap", ".png")

        feature_cpu = feature_tensor.detach().cpu()
        torch.save(feature_cpu, tensor_path)

        preview = feature_tensor.detach().mean(dim=1, keepdim=True)
        preview = preview - preview.amin(dim=(2, 3), keepdim=True)
        preview = preview / (preview.amax(dim=(2, 3), keepdim=True) + 1e-8)
        utils.save_image(preview.cpu(), str(preview_path))

        return tensor_path, self._to_output_url(tensor_path), preview_path, self._to_output_url(preview_path)

    def _save_latent_bundle(
        self,
        w: torch.Tensor,
        noises: Optional[List[torch.Tensor]],
        session_id: str,
        label: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, str]:
        path = self._next_path(self.output_dir, session_id, f"{label}_latent", ".pt")
        bundle: Dict[str, Any] = {
            "w": w.detach().cpu(),
            "noise": [noise.detach().cpu() for noise in noises] if noises is not None else None,
        }
        if metadata:
            bundle.update(metadata)
        torch.save(bundle, path)
        return path, self._to_output_url(path)

    def _store_session(self, session: DragSession) -> None:
        with self._sessions_lock:
            self._sessions[session.session_id] = session

    def _get_session(self, session_id: str) -> DragSession:
        with self._sessions_lock:
            if session_id not in self._sessions:
                raise ValueError(f"Unknown session_id: {session_id}")
            return self._sessions[session_id]

    def _load_image_for_projection(self, image_path: Path) -> torch.Tensor:
        pil_image = Image.open(image_path).convert("RGB")

        resize_kwargs: Dict[str, Any] = {}
        if hasattr(transforms, "InterpolationMode"):
            resize_kwargs["interpolation"] = transforms.InterpolationMode.LANCZOS

        try:
            resize_transform = transforms.Resize(self.g_ema.size, antialias=True, **resize_kwargs)
        except TypeError:
            resize_transform = transforms.Resize(self.g_ema.size, **resize_kwargs)

        transform = transforms.Compose(
            [
                resize_transform,
                transforms.CenterCrop(self.g_ema.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        image_tensor = transform(pil_image).unsqueeze(0)
        return image_tensor.to(self.device)

    def _project_image_to_w(
        self,
        target_image: torch.Tensor,
        steps: int,
        lr: float,
        noise_regularize_weight: float,
        fullres_weight: float,
        lowres_weight: float,
        gradient_weight: float,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, float]:
        if steps < 1:
            raise ValueError("Projection steps must be at least 1")

        device = target_image.device
        batch_size = target_image.shape[0]

        with torch.no_grad():
            mean_latent_samples = torch.randn(4096, 512, device=device)
            latent_out = self.g_ema.style(mean_latent_samples)
            mean_w = latent_out.mean(0)
            latent_std = ((latent_out - mean_w).pow(2).sum() / mean_latent_samples.shape[0]).sqrt()

        latent_in = mean_w.detach().clone().unsqueeze(0).repeat(batch_size, 1)
        latent_in = latent_in.unsqueeze(1).repeat(1, self.g_ema.n_latent, 1)
        latent_in.requires_grad = True

        noises = []
        for noise in self.g_ema.make_noise():
            seeded_noise = noise.repeat(batch_size, 1, 1, 1).normal_()
            seeded_noise.requires_grad = True
            noises.append(seeded_noise)

        optimizer = optim.Adam([latent_in] + noises, lr=lr)

        target_256 = F.adaptive_avg_pool2d(target_image, (256, 256))
        target_grad_x, target_grad_y = image_gradients(target_image)
        best_recon = float("inf")
        best_w = latent_in.detach().clone()
        best_noises = [noise.detach().clone() for noise in noises]

        for step in range(steps):
            t = step / steps
            current_lr = get_lr(t, lr)
            optimizer.param_groups[0]["lr"] = current_lr

            noise_strength = latent_std * 0.05 * max(0.0, 1.0 - t / 0.75) ** 2
            latent_n = latent_noise(latent_in, float(noise_strength.item()))

            image_gen, _ = self.g_ema([latent_n], input_is_latent=True, noise=noises, randomize_noise=False)
            image_gen_256 = F.adaptive_avg_pool2d(image_gen, (256, 256))
            gen_grad_x, gen_grad_y = image_gradients(image_gen)

            fullres_mse_loss = F.mse_loss(image_gen, target_image)
            lowres_mse_loss = F.mse_loss(image_gen_256, target_256)
            gradient_loss = F.l1_loss(gen_grad_x, target_grad_x) + F.l1_loss(gen_grad_y, target_grad_y)

            recon_loss = (
                fullres_weight * fullres_mse_loss
                + lowres_weight * lowres_mse_loss
                + gradient_weight * gradient_loss
            )
            noise_loss = noise_regularize(noises)
            loss = recon_loss + noise_regularize_weight * noise_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            noise_normalize_(noises)

            recon_value = recon_loss.item()
            if recon_value < best_recon:
                best_recon = recon_value
                best_w = latent_in.detach().clone()
                best_noises = [noise.detach().clone() for noise in noises]

        with torch.no_grad():
            projected_image, _ = self.g_ema(
                [best_w],
                input_is_latent=True,
                noise=best_noises,
                randomize_noise=False,
            )

        return best_w.detach(), [noise.detach() for noise in best_noises], projected_image.detach(), best_recon

    def _validate_points(self, points: Sequence[Sequence[int]], feature_h: int, feature_w: int, label: str) -> None:
        for point_idx, point in enumerate(points):
            if len(point) != 2:
                raise ValueError(f"{label} point #{point_idx + 1} must contain exactly [Y, X].")

            y, x = int(point[0]), int(point[1])
            if y < 0 or y >= feature_h or x < 0 or x >= feature_w:
                raise ValueError(
                    f"{label} point #{point_idx + 1} [{y}, {x}] is out of bounds for feature map "
                    f"size [{feature_h}, {feature_w}]."
                )

    def _point_to_feature(
        self,
        point: Dict[str, Any],
        image_h: int,
        image_w: int,
        feature_h: int,
        feature_w: int,
    ) -> Tuple[int, int]:
        if not isinstance(point, dict):
            raise ValueError("Point must be an object with x and y fields.")

        if "x" not in point or "y" not in point:
            raise ValueError("Point must include both x and y.")

        x = float(point["x"])
        y = float(point["y"])

        mapped_y = int(round((y / max(1, image_h - 1)) * max(1, feature_h - 1)))
        mapped_x = int(round((x / max(1, image_w - 1)) * max(1, feature_w - 1)))
        return clamp_point(mapped_y, mapped_x, feature_h, feature_w)

    def _pairs_to_feature_points(
        self,
        pairs: Sequence[Dict[str, Any]],
        image_h: int,
        image_w: int,
        feature_h: int,
        feature_w: int,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        handle_points: List[List[int]] = []
        target_points: List[List[int]] = []

        if len(pairs) == 0:
            raise ValueError("At least one handle-target pair is required.")

        for pair in pairs:
            if not isinstance(pair, dict):
                raise ValueError("Each pair must be an object with handle and target points.")

            handle = self._point_to_feature(pair.get("handle", {}), image_h, image_w, feature_h, feature_w)
            target = self._point_to_feature(pair.get("target", {}), image_h, image_w, feature_h, feature_w)

            handle_points.append([handle[0], handle[1]])
            target_points.append([target[0], target[1]])

        return handle_points, target_points

    def _feature_to_image_point(
        self,
        point_y: int,
        point_x: int,
        image_h: int,
        image_w: int,
        feature_h: int,
        feature_w: int,
    ) -> Dict[str, int]:
        image_y = int(round((point_y / max(1, feature_h - 1)) * max(1, image_h - 1)))
        image_x = int(round((point_x / max(1, feature_w - 1)) * max(1, image_w - 1)))
        return {"x": image_x, "y": image_y}

    def create_generated_session(self, seed: Optional[int], generation_steps: int) -> Dict[str, Any]:
        if generation_steps < 1:
            raise ValueError("generation_steps must be at least 1.")

        session_id = self._new_session_id()

        if seed is None:
            seed = int.from_bytes(os.urandom(8), "big") & 0x7FFFFFFF

        with self._gpu_lock:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))

            sample_z = torch.randn((1, 512), generator=generator, device=self.device)
            for _ in range(generation_steps - 1):
                jitter = torch.randn((1, 512), generator=generator, device=self.device)
                sample_z = 0.85 * sample_z + 0.15 * jitter

            with torch.no_grad():
                w = self.g_ema.style(sample_z)
                image, features = self.g_ema(
                    [w],
                    input_is_latent=True,
                    return_features=True,
                    randomize_noise=False,
                )

            if features is None:
                raise RuntimeError("Failed to extract feature map for generated image.")

            _, initial_image_url = self._save_image(image, session_id, "initial_image")
            _, initial_feature_tensor_url, _, initial_feature_map_url = self._save_feature_map_outputs(
                features,
                session_id,
                "initial",
            )
            _, latent_url = self._save_latent_bundle(
                w,
                None,
                session_id,
                "initial",
                metadata={"seed": int(seed), "generation_steps": int(generation_steps), "mode": "generate"},
            )

            image_h, image_w = int(image.shape[2]), int(image.shape[3])
            feature_h, feature_w = int(features.shape[2]), int(features.shape[3])

            session = DragSession(
                session_id=session_id,
                w=w.detach(),
                noises=None,
                image_height=image_h,
                image_width=image_w,
                feature_height=feature_h,
                feature_width=feature_w,
                mode="generate",
                seed=int(seed),
                generation_steps=int(generation_steps),
                source_image_url=None,
                initial_image_url=initial_image_url,
                initial_feature_map_url=initial_feature_map_url,
                initial_feature_tensor_url=initial_feature_tensor_url,
            )
            self._store_session(session)

        return {
            "session_id": session_id,
            "mode": "generate",
            "seed": int(seed),
            "generation_steps": int(generation_steps),
            "source_image_url": None,
            "initial_image_url": initial_image_url,
            "initial_feature_map_url": initial_feature_map_url,
            "initial_feature_tensor_url": initial_feature_tensor_url,
            "initial_latent_url": latent_url,
            "image_size": {"height": image_h, "width": image_w},
            "feature_size": {"height": feature_h, "width": feature_w},
            "device": self.device,
        }

    def create_inversion_session(
        self,
        image_bytes: bytes,
        filename: str,
        steps: int,
        lr: float,
        noise_regularize_weight: float,
        fullres_weight: float,
        lowres_weight: float,
        gradient_weight: float,
    ) -> Dict[str, Any]:
        if steps < 1:
            raise ValueError("Inversion steps must be at least 1.")
        if lr <= 0:
            raise ValueError("Inversion learning rate must be positive.")
        if noise_regularize_weight < 0:
            raise ValueError("noise_regularize_weight must be non-negative.")
        if fullres_weight <= 0:
            raise ValueError("fullres_weight must be positive.")
        if lowres_weight < 0:
            raise ValueError("lowres_weight must be non-negative.")
        if gradient_weight < 0:
            raise ValueError("gradient_weight must be non-negative.")

        if len(image_bytes) == 0:
            raise ValueError("Uploaded image is empty.")

        session_id = self._new_session_id()
        suffix = Path(filename).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            suffix = ".png"

        upload_path = self._next_path(self.upload_dir, session_id, "source", suffix)
        upload_path.write_bytes(image_bytes)

        try:
            with Image.open(upload_path) as img:
                img.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Invalid image upload: {exc}") from exc

        with self._gpu_lock:
            target_image = self._load_image_for_projection(upload_path)

            projected_w, projected_noises, _, projection_loss = self._project_image_to_w(
                target_image=target_image,
                steps=steps,
                lr=lr,
                noise_regularize_weight=noise_regularize_weight,
                fullres_weight=fullres_weight,
                lowres_weight=lowres_weight,
                gradient_weight=gradient_weight,
            )

            with torch.no_grad():
                initial_image, initial_features = self.g_ema(
                    [projected_w],
                    input_is_latent=True,
                    return_features=True,
                    noise=projected_noises,
                    randomize_noise=False,
                )

            if initial_features is None:
                raise RuntimeError("Failed to extract feature map from projected image.")

            _, initial_image_url = self._save_image(initial_image, session_id, "initial_image")
            _, initial_feature_tensor_url, _, initial_feature_map_url = self._save_feature_map_outputs(
                initial_features,
                session_id,
                "initial",
            )
            _, latent_url = self._save_latent_bundle(
                projected_w,
                projected_noises,
                session_id,
                "initial",
                metadata={"mode": "invert", "projection_steps": int(steps), "projection_loss": float(projection_loss)},
            )

            image_h, image_w = int(initial_image.shape[2]), int(initial_image.shape[3])
            feature_h, feature_w = int(initial_features.shape[2]), int(initial_features.shape[3])

            session = DragSession(
                session_id=session_id,
                w=projected_w.detach(),
                noises=[noise.detach() for noise in projected_noises],
                image_height=image_h,
                image_width=image_w,
                feature_height=feature_h,
                feature_width=feature_w,
                mode="invert",
                seed=None,
                generation_steps=None,
                source_image_url=self._to_upload_url(upload_path),
                initial_image_url=initial_image_url,
                initial_feature_map_url=initial_feature_map_url,
                initial_feature_tensor_url=initial_feature_tensor_url,
            )
            self._store_session(session)

        return {
            "session_id": session_id,
            "mode": "invert",
            "projection_steps": int(steps),
            "projection_loss": float(projection_loss),
            "source_image_url": self._to_upload_url(upload_path),
            "initial_image_url": initial_image_url,
            "initial_feature_map_url": initial_feature_map_url,
            "initial_feature_tensor_url": initial_feature_tensor_url,
            "initial_latent_url": latent_url,
            "image_size": {"height": image_h, "width": image_w},
            "feature_size": {"height": feature_h, "width": feature_w},
            "device": self.device,
        }

    def run_drag(
        self,
        session_id: str,
        pairs: Sequence[Dict[str, Any]],
        drag_steps: int,
        drag_lr: float,
    ) -> Dict[str, Any]:
        if drag_steps < 1:
            raise ValueError("drag_steps must be at least 1.")
        if drag_lr <= 0:
            raise ValueError("drag_lr must be positive.")

        session = self._get_session(session_id)

        with self._gpu_lock:
            w_opt = session.w.detach().clone()
            w_opt.requires_grad = True
            optimizer = optim.Adam([w_opt], lr=drag_lr)

            handle_points, target_points = self._pairs_to_feature_points(
                pairs,
                session.image_height,
                session.image_width,
                session.feature_height,
                session.feature_width,
            )
            self._validate_points(handle_points, session.feature_height, session.feature_width, "Handle")
            self._validate_points(target_points, session.feature_height, session.feature_width, "Target")

            for _ in range(drag_steps):
                optimizer.zero_grad()

                image, features = self.g_ema(
                    [w_opt],
                    input_is_latent=True,
                    return_features=True,
                    noise=session.noises,
                    randomize_noise=False,
                )

                if features is None:
                    raise RuntimeError("Failed to extract feature map during drag optimization.")

                feature_h, feature_w = features.shape[2], features.shape[3]
                point_losses: List[torch.Tensor] = []
                original_features: List[torch.Tensor] = []

                for pair_idx, (handle_point, target_point) in enumerate(zip(handle_points, target_points)):
                    hy, hx = clamp_point(handle_point[0], handle_point[1], feature_h, feature_w)
                    ty, tx = clamp_point(target_point[0], target_point[1], feature_h, feature_w)
                    handle_points[pair_idx] = [hy, hx]
                    target_points[pair_idx] = [ty, tx]

                    f_original = features[:, :, hy, hx].detach()

                    direction_y = int(torch.sign(torch.tensor(float(ty - hy), device=features.device)).item())
                    direction_x = int(torch.sign(torch.tensor(float(tx - hx), device=features.device)).item())

                    step_y, step_x = clamp_point(hy + direction_y, hx + direction_x, feature_h, feature_w)
                    f_shifted = features[:, :, step_y, step_x]

                    point_losses.append(F.l1_loss(f_shifted, f_original))
                    original_features.append(f_original)

                if len(point_losses) == 0:
                    raise ValueError("No valid handle-target pairs were provided.")

                loss = torch.stack(point_losses).mean()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    features_norm = features / (features.norm(dim=1, keepdim=True) + 1e-8)
                    updated_handle_points = []

                    for f_original in original_features:
                        f_original_norm = f_original / (f_original.norm(dim=1, keepdim=True) + 1e-8)
                        similarity = torch.einsum("nc,nchw->nhw", f_original_norm, features_norm)
                        max_sim_idx = torch.argmax(similarity)
                        new_hy, new_hx = divmod(max_sim_idx.item(), similarity.shape[2])
                        updated_handle_points.append([new_hy, new_hx])

                    handle_points = updated_handle_points

            with torch.no_grad():
                final_image, final_features = self.g_ema(
                    [w_opt.detach()],
                    input_is_latent=True,
                    return_features=True,
                    noise=session.noises,
                    randomize_noise=False,
                )

            if final_features is None:
                raise RuntimeError("Failed to extract final feature map.")

            session.w = w_opt.detach()
            _, output_image_url = self._save_image(final_image, session.session_id, "output_image")
            _, output_feature_tensor_url, _, output_feature_map_url = self._save_feature_map_outputs(
                final_features,
                session.session_id,
                "output",
            )
            _, output_latent_url = self._save_latent_bundle(
                session.w,
                session.noises,
                session.session_id,
                "output",
                metadata={"mode": session.mode, "drag_steps": int(drag_steps), "drag_lr": float(drag_lr)},
            )

            session.last_output_image_url = output_image_url
            session.last_output_feature_map_url = output_feature_map_url
            session.last_output_feature_tensor_url = output_feature_tensor_url

            updated_handles_image = [
                self._feature_to_image_point(
                    point_y=point[0],
                    point_x=point[1],
                    image_h=session.image_height,
                    image_w=session.image_width,
                    feature_h=session.feature_height,
                    feature_w=session.feature_width,
                )
                for point in handle_points
            ]

        return {
            "session_id": session.session_id,
            "drag_steps": int(drag_steps),
            "drag_lr": float(drag_lr),
            "output_image_url": output_image_url,
            "output_feature_map_url": output_feature_map_url,
            "output_feature_tensor_url": output_feature_tensor_url,
            "output_latent_url": output_latent_url,
            "updated_handles_image": updated_handles_image,
        }

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        session = self._get_session(session_id)
        return {
            "session_id": session.session_id,
            "mode": session.mode,
            "seed": session.seed,
            "generation_steps": session.generation_steps,
            "source_image_url": session.source_image_url,
            "initial_image_url": session.initial_image_url,
            "initial_feature_map_url": session.initial_feature_map_url,
            "initial_feature_tensor_url": session.initial_feature_tensor_url,
            "last_output_image_url": session.last_output_image_url,
            "last_output_feature_map_url": session.last_output_feature_map_url,
            "last_output_feature_tensor_url": session.last_output_feature_tensor_url,
            "image_size": {"height": session.image_height, "width": session.image_width},
            "feature_size": {"height": session.feature_height, "width": session.feature_width},
            "device": self.device,
        }
