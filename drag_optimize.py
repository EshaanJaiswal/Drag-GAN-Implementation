import torch
from torch import optim
from torch.nn import functional as F
from model import Generator
from torchvision import utils, transforms
import argparse
import sys
import os
import math
import hashlib
from PIL import Image


PROJECTION_CACHE_VERSION = "v3_wplus_sharp"


def save_image_compat(tensor, path):
    # torchvision changed `range` to `value_range`; support both.
    try:
        utils.save_image(tensor, path, normalize=True, value_range=(-1, 1))
    except TypeError:
        utils.save_image(tensor, path, normalize=True, range=(-1, 1))


def get_points_via_console(num_points=0):
    print("Falling back to terminal input for points.")
    handle_points = []
    target_points = []

    try:
        if num_points == 0:
            count_raw = input("How many handle-target pairs do you want to enter?\n> ").strip()
            num_points = int(count_raw)
            if num_points < 1:
                raise ValueError("num_points must be >= 1")

        for point_idx in range(num_points):
            handle_raw = input(f"Enter handle point #{point_idx + 1} as: Y X\n> ").strip()
            target_raw = input(f"Enter target point #{point_idx + 1} as: Y X\n> ").strip()
            hy, hx = map(int, handle_raw.split())
            ty, tx = map(int, target_raw.split())
            handle_points.append([hy, hx])
            target_points.append([ty, tx])

        return handle_points, target_points
    except Exception:
        print("Invalid input. Use repeated --handle Y X and --target Y X")
        sys.exit(1)


def choose_point_picker_image(mode, existing_path, generated_path="drag_step_00.png"):
    mode = (mode or "ask").lower()

    if mode == "generate":
        return "generate", generated_path

    if mode == "existing":
        if not os.path.exists(existing_path):
            print(f"Error: Existing image not found: {existing_path}")
            sys.exit(1)
        return "existing", existing_path

    if not sys.stdin.isatty():
        print("Non-interactive input detected. Defaulting to generated image.")
        return "generate", generated_path

    print("Choose image source for drag-point selection:")
    print("  1) Generate a fresh image")
    print(f"  2) Use existing image (default: {existing_path})")
    choice = input("> ").strip().lower()

    if choice in ("", "1", "g", "generate"):
        return "generate", generated_path

    if choice in ("2", "e", "existing"):
        entered = input(f"Existing image path [{existing_path}]: ").strip()
        selected_path = entered if entered else existing_path
        if not os.path.exists(selected_path):
            print(f"Error: Existing image not found: {selected_path}")
            sys.exit(1)
        return "existing", selected_path

    print("Invalid choice. Use 1/2 or generate/existing.")
    sys.exit(1)


def latent_sidecar_path(image_path):
    stem, _ = os.path.splitext(image_path)
    return f"{stem}_latent.pt"


def extract_latent_tensor(latent_data):
    if torch.is_tensor(latent_data):
        return latent_data

    if isinstance(latent_data, dict):
        if "w" in latent_data and torch.is_tensor(latent_data["w"]):
            return latent_data["w"]

        if "latent" in latent_data and torch.is_tensor(latent_data["latent"]):
            return latent_data["latent"]

        for value in latent_data.values():
            if isinstance(value, dict):
                if "w" in value and torch.is_tensor(value["w"]):
                    return value["w"]

                if "latent" in value and torch.is_tensor(value["latent"]):
                    return value["latent"]

    raise ValueError("Could not find a latent tensor in the provided file.")


def extract_noise_tensors(latent_data):
    if isinstance(latent_data, dict):
        if "noise" in latent_data and isinstance(latent_data["noise"], (list, tuple)):
            if all(torch.is_tensor(noise) for noise in latent_data["noise"]):
                return list(latent_data["noise"])

        for value in latent_data.values():
            if isinstance(value, dict):
                if "noise" in value and isinstance(value["noise"], (list, tuple)):
                    if all(torch.is_tensor(noise) for noise in value["noise"]):
                        return list(value["noise"])

    return None


def prepare_w_latent(w_tensor, g_ema, device):
    if w_tensor.ndim == 1:
        w_tensor = w_tensor.unsqueeze(0)
    elif w_tensor.ndim == 2 and w_tensor.shape[0] == g_ema.n_latent and w_tensor.shape[1] == 512:
        w_tensor = w_tensor.unsqueeze(0)

    if w_tensor.ndim not in (2, 3):
        print(f"Error: Unexpected latent shape {tuple(w_tensor.shape)}")
        sys.exit(1)

    if w_tensor.shape[0] != 1:
        print(f"Error: Expected a single-image latent, got batch size {w_tensor.shape[0]}")
        sys.exit(1)

    return w_tensor.to(device=device, dtype=torch.float32)


def prepare_projection_noises(noise_tensors, g_ema, device):
    if noise_tensors is None:
        return None

    if len(noise_tensors) != g_ema.num_layers:
        print(
            f"Warning: Ignoring loaded noise list because it has {len(noise_tensors)} layers; "
            f"expected {g_ema.num_layers}."
        )
        return None

    prepared_noises = []
    for noise_idx, noise in enumerate(noise_tensors):
        if not torch.is_tensor(noise):
            print(f"Warning: Ignoring loaded noise at index {noise_idx} because it is not a tensor.")
            return None

        if noise.ndim == 3:
            noise = noise.unsqueeze(0)

        if noise.ndim != 4:
            print(f"Warning: Ignoring loaded noise at index {noise_idx} with shape {tuple(noise.shape)}.")
            return None

        if noise.shape[0] != 1:
            noise = noise[:1]

        prepared_noises.append(noise.to(device=device, dtype=torch.float32).detach())

    return prepared_noises


def noise_regularize(noises):
    loss = 0.0

    for noise in noises:
        size = noise.shape[2]
        current_noise = noise

        while True:
            loss = (
                loss
                + (current_noise * torch.roll(current_noise, shifts=1, dims=3)).mean().pow(2)
                + (current_noise * torch.roll(current_noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            current_noise = current_noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            current_noise = current_noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()
        noise.data.add_(-mean).div_(std + 1e-8)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength
    return latent + noise


def image_gradients(image):
    grad_x = image[:, :, :, 1:] - image[:, :, :, :-1]
    grad_y = image[:, :, 1:, :] - image[:, :, :-1, :]
    return grad_x, grad_y


def get_image_file_signature(image_path):
    resolved_image_path = os.path.abspath(os.path.expanduser(image_path))
    image_stat = os.stat(resolved_image_path)
    return resolved_image_path, int(image_stat.st_size), int(image_stat.st_mtime_ns)


def get_projection_cache_paths(image_path, cache_dir):
    resolved_image_path, image_size, image_mtime_ns = get_image_file_signature(image_path)
    cache_root = os.path.abspath(os.path.expanduser(cache_dir))
    os.makedirs(cache_root, exist_ok=True)

    image_stem = os.path.splitext(os.path.basename(resolved_image_path))[0]
    cache_key = f"{resolved_image_path}|{image_size}|{image_mtime_ns}|{PROJECTION_CACHE_VERSION}"
    image_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:10]
    prefix = os.path.join(cache_root, f"{image_stem}_{image_hash}")

    target_path = f"{prefix}_target.png"
    projected_path = f"{prefix}_projected.png"
    latent_path = f"{prefix}_latent.pt"

    return target_path, projected_path, latent_path


def load_image_for_projection(image_path, output_size, device):
    pil_image = Image.open(image_path).convert("RGB")

    if pil_image.width < output_size or pil_image.height < output_size:
        print(
            f"Warning: input image is {pil_image.width}x{pil_image.height}; "
            f"upscaling to {output_size}x{output_size} can reduce sharpness."
        )

    resize_kwargs = {}
    if hasattr(transforms, "InterpolationMode"):
        resize_kwargs["interpolation"] = transforms.InterpolationMode.LANCZOS

    try:
        resize_transform = transforms.Resize(output_size, antialias=True, **resize_kwargs)
    except TypeError:
        resize_transform = transforms.Resize(output_size, **resize_kwargs)

    transform = transforms.Compose(
        [
            resize_transform,
            transforms.CenterCrop(output_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    image_tensor = transform(pil_image).unsqueeze(0)
    return image_tensor.to(device)


def project_image_to_w(
    g_ema,
    target_image,
    steps,
    lr,
    noise_regularize_weight=3e4,
    fullres_weight=1.0,
    lowres_weight=0.25,
    gradient_weight=0.35,
):
    if steps < 1:
        raise ValueError("Projection steps must be at least 1")

    device = target_image.device
    batch_size = target_image.shape[0]

    with torch.no_grad():
        mean_latent_samples = torch.randn(4096, 512, device=device)
        latent_out = g_ema.style(mean_latent_samples)
        mean_w = latent_out.mean(0)
        latent_std = ((latent_out - mean_w).pow(2).sum() / mean_latent_samples.shape[0]).sqrt()

    latent_in = mean_w.detach().clone().unsqueeze(0).repeat(batch_size, 1)
    latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
    latent_in.requires_grad = True

    noises = []
    for noise in g_ema.make_noise():
        noise = noise.repeat(batch_size, 1, 1, 1).normal_()
        noise.requires_grad = True
        noises.append(noise)

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

        noise_strength = latent_std * 0.05 * max(0, 1 - t / 0.75) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())

        image_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises, randomize_noise=False)
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

        if step % 50 == 0 or step == steps - 1:
            print(
                f"Projection step {step + 1}/{steps} | Recon: {recon_value:.6f} | "
                f"MSE1024: {fullres_mse_loss.item():.6f} | "
                f"MSE256: {lowres_mse_loss.item():.6f} | "
                f"Grad: {gradient_loss.item():.6f} | NoiseReg: {noise_loss.item():.6f}"
            )

    with torch.no_grad():
        projected_image, _ = g_ema(
            [best_w],
            input_is_latent=True,
            noise=best_noises,
            randomize_noise=False,
        )

    return best_w, best_noises, projected_image.detach(), best_recon


def validate_points(points, feature_h, feature_w, label):
    for point_idx, point in enumerate(points):
        if len(point) != 2:
            print(f"Error: {label} point #{point_idx + 1} must contain exactly 2 values [Y, X].")
            sys.exit(1)

        y, x = int(point[0]), int(point[1])
        if y < 0 or y >= feature_h or x < 0 or x >= feature_w:
            print(
                f"Error: {label} point #{point_idx + 1} [{y}, {x}] is out of bounds for feature map "
                f"size [{feature_h}, {feature_w}]."
            )
            sys.exit(1)


def clamp_point(y, x, feature_h, feature_w):
    clamped_y = max(0, min(int(y), feature_h - 1))
    clamped_x = max(0, min(int(x), feature_w - 1))
    return clamped_y, clamped_x

# 1. Interactive UI Function
def get_points_via_ui(image_path='drag_step_00.png', num_points=0):
    print("No coordinates provided via CLI. Launching interactive picker...")
    try:
        import matplotlib
        # Try to switch off non-interactive backend when possible.
        if matplotlib.get_backend().lower() == 'agg':
            for candidate in ("TkAgg", "QtAgg", "GTK3Agg", "WXAgg"):
                try:
                    matplotlib.use(candidate, force=True)
                    break
                except Exception:
                    continue

        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        if matplotlib.get_backend().lower() == 'agg':
            raise RuntimeError("Matplotlib is using non-interactive backend 'Agg'.")

        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError("No display server detected in environment.")

        img = mpimg.imread(image_path)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)

        if num_points == 0:
            ax.set_title(
                "Manual mode: click Handle then Target pairs, press Enter when done"
            )
            coords = plt.ginput(-1, timeout=-1)
        else:
            ax.set_title(
                f"Pick {num_points} pair(s): Handle then Target for each pair (Close window when done)"
            )
            coords = plt.ginput(2 * num_points, timeout=-1)

        plt.close()

        if len(coords) < 2:
            print("Error: At least one handle-target pair is required. Exiting.")
            sys.exit(1)

        if len(coords) % 2 != 0:
            print("Error: You must provide an even number of clicks (handle/target pairs). Exiting.")
            sys.exit(1)

        if num_points == 0:
            num_points = len(coords) // 2
        elif len(coords) != 2 * num_points:
            print(f"Error: You must click exactly {2 * num_points} points. Exiting.")
            sys.exit(1)

        handle_points = []
        target_points = []
        for point_idx in range(num_points):
            handle_xy = coords[2 * point_idx]
            target_xy = coords[2 * point_idx + 1]

            hy, hx = int(handle_xy[1] / 4), int(handle_xy[0] / 4)
            ty, tx = int(target_xy[1] / 4), int(target_xy[0] / 4)

            handle_points.append([hy, hx])
            target_points.append([ty, tx])

        return handle_points, target_points
    except Exception as e:
        print(f"\n[!] UI Failed: {e}")
        print("[!] Run with repeated arguments instead: python drag_optimize.py --handle Y X --target Y X [...]")
        print("[!] Trying terminal input fallback now.\n")
        return get_points_via_console(num_points=num_points)

# 2. Setup Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--handle",
    type=int,
    nargs=2,
    action="append",
    default=None,
    help="Handle point in Y X format. Repeat for multiple points.",
)
parser.add_argument(
    "--target",
    type=int,
    nargs=2,
    action="append",
    default=None,
    help="Target point in Y X format. Repeat for multiple points.",
)
parser.add_argument(
    "--num-points",
    type=int,
    default=0,
    help="Number of handle-target pairs to collect in UI/console mode. Use 0 for manual finish mode.",
)
parser.add_argument(
    "--image-mode",
    type=str,
    choices=["ask", "generate", "existing"],
    default="ask",
    help="Image source for point selection when handle/target are not provided.",
)
parser.add_argument(
    "--existing-image",
    type=str,
    default="drag_step_00.png",
    help="Path to existing image when using --image-mode existing.",
)
parser.add_argument(
    "--existing-latent",
    type=str,
    default=None,
    help="Optional latent file for existing image mode. If missing, the script can auto-project the image.",
)
parser.add_argument(
    "--picker-image",
    type=str,
    choices=["source", "projected"],
    default="source",
    help="Image shown for point selection in existing mode.",
)
parser.add_argument(
    "--project-steps",
    type=int,
    default=900,
    help="Projection steps used when auto-projecting an existing image without a latent.",
)
parser.add_argument(
    "--project-lr",
    type=float,
    default=0.05,
    help="Projection learning rate for auto-projecting an existing image.",
)
parser.add_argument(
    "--project-noise-reg",
    type=float,
    default=3e4,
    help="Weight for projection noise regularization. Lower values can preserve sharper details.",
)
parser.add_argument(
    "--project-fullres-weight",
    type=float,
    default=1.0,
    help="Weight for full-resolution projection MSE loss.",
)
parser.add_argument(
    "--project-lowres-weight",
    type=float,
    default=0.25,
    help="Weight for low-resolution (256px) projection MSE loss.",
)
parser.add_argument(
    "--project-gradient-weight",
    type=float,
    default=0.35,
    help="Weight for image-gradient loss used to reduce blur.",
)
parser.add_argument(
    "--project-cache-dir",
    type=str,
    default=".drag_projection_cache",
    help="Directory used to cache projected images and latents for external input images.",
)
parser.add_argument(
    "--force-reproject",
    action="store_true",
    help="Force re-projecting an existing image even when a cached latent already exists.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Optional fixed seed for reproducible generation. By default, a random seed is used.",
)
args = parser.parse_args()

if args.num_points < 0:
    print("Error: --num-points must be 0 or greater.")
    sys.exit(1)

if args.project_steps < 1:
    print("Error: --project-steps must be at least 1.")
    sys.exit(1)

if args.project_lr <= 0:
    print("Error: --project-lr must be positive.")
    sys.exit(1)

if args.project_noise_reg < 0:
    print("Error: --project-noise-reg must be non-negative.")
    sys.exit(1)

if args.project_fullres_weight <= 0:
    print("Error: --project-fullres-weight must be positive.")
    sys.exit(1)

if args.project_lowres_weight < 0:
    print("Error: --project-lowres-weight must be non-negative.")
    sys.exit(1)

if args.project_gradient_weight < 0:
    print("Error: --project-gradient-weight must be non-negative.")
    sys.exit(1)

cli_handle_points = args.handle or []
cli_target_points = args.target or []

if (len(cli_handle_points) == 0) != (len(cli_target_points) == 0):
    print("Error: Provide both --handle and --target. Repeat each argument once per pair.")
    sys.exit(1)

if len(cli_handle_points) != len(cli_target_points):
    print("Error: Number of --handle entries must match number of --target entries.")
    sys.exit(1)

needs_ui_points = len(cli_handle_points) == 0
if needs_ui_points:
    num_points = args.num_points
else:
    num_points = len(cli_handle_points)
    if args.num_points not in (0, num_points):
        print(
            f"Info: Ignoring --num-points={args.num_points} because {num_points} CLI point pair(s) were provided."
        )

device = 'cuda'
if args.seed is None:
    run_seed = int.from_bytes(os.urandom(8), "big") & 0x7FFFFFFF
else:
    run_seed = int(args.seed)
torch.manual_seed(run_seed)
print(f"Using seed: {run_seed}")

g_ema = Generator(1024, 512, 8).to(device)
checkpoint = torch.load('ffhq.pt') 
g_ema.load_state_dict(checkpoint['g_ema'])
g_ema.eval() 
point_picker_image = "drag_step_00.png"

if args.image_mode == "ask" and needs_ui_points:
    image_mode, point_picker_image = choose_point_picker_image(
        mode=args.image_mode,
        existing_path=args.existing_image,
        generated_path="drag_step_00.png",
    )
elif args.image_mode == "existing":
    image_mode = "existing"
    point_picker_image = args.existing_image
    if not os.path.exists(point_picker_image):
        print(f"Error: Existing image not found: {point_picker_image}")
        sys.exit(1)
else:
    image_mode = "generate"

w_init = None
projection_noises = None

# 3. Generate and save the initial image BEFORE picking points
if image_mode == "generate":
    sample_z = torch.randn(1, 512, device=device)
    with torch.no_grad():
        w_init = g_ema.style(sample_z)
        initial_image, _ = g_ema([w_init], input_is_latent=True, return_features=True, randomize_noise=False)
        save_image_compat(initial_image, "drag_step_00.png")

    generated_latent_path = latent_sidecar_path("drag_step_00.png")
    torch.save({"w": w_init.detach().cpu(), "seed": run_seed}, generated_latent_path)
    print(f"Saved latent: {generated_latent_path}")

    point_picker_image = "drag_step_00.png"
else:
    source_image_path = point_picker_image
    print(f"Using existing image for point selection: {source_image_path}")

    try:
        source_image_path_resolved, source_image_size, source_image_mtime_ns = get_image_file_signature(
            source_image_path
        )
    except OSError as e:
        print(f"Error reading existing image metadata {source_image_path}: {e}")
        sys.exit(1)

    default_sidecar_path = latent_sidecar_path(source_image_path)
    cached_target_path, cached_projected_path, cached_latent_path = get_projection_cache_paths(
        source_image_path,
        args.project_cache_dir,
    )

    candidate_latent_paths = []
    if args.existing_latent:
        candidate_latent_paths.append(args.existing_latent)
    else:
        candidate_latent_paths.append(cached_latent_path)
        if os.path.exists(default_sidecar_path):
            try:
                default_sidecar_mtime_ns = int(os.stat(default_sidecar_path).st_mtime_ns)
                if default_sidecar_mtime_ns >= source_image_mtime_ns:
                    candidate_latent_paths.append(default_sidecar_path)
                else:
                    print(
                        f"Skipping stale sidecar latent: {default_sidecar_path}. "
                        "Use --existing-latent to force it."
                    )
            except OSError as e:
                print(f"Warning: Could not read sidecar latent metadata {default_sidecar_path}: {e}")

    chosen_latent_path = None
    if not args.force_reproject:
        for candidate_path in candidate_latent_paths:
            if os.path.exists(candidate_path):
                chosen_latent_path = candidate_path
                break

    if chosen_latent_path is not None:
        try:
            loaded = torch.load(chosen_latent_path, map_location="cpu")

            if isinstance(loaded, dict) and "projection_version" in loaded:
                loaded_projection_version = loaded.get("projection_version")
                if loaded_projection_version != PROJECTION_CACHE_VERSION:
                    print(
                        "Cached latent was created with an older projection version. "
                        "Re-run with --force-reproject for sharper results."
                    )

            if chosen_latent_path == default_sidecar_path and args.existing_latent is None:
                print(
                    "Using legacy sidecar latent fallback. "
                    "If output looks blurry, run once with --force-reproject."
                )

            w_init = extract_latent_tensor(loaded)
            loaded_noises = extract_noise_tensors(loaded)
            projection_noises = prepare_projection_noises(loaded_noises, g_ema, device)
        except Exception as e:
            print(f"Error loading latent file {chosen_latent_path}: {e}")
            sys.exit(1)

        print(f"Loaded latent: {chosen_latent_path}")

        if (
            args.picker_image == "projected"
            and chosen_latent_path == cached_latent_path
            and os.path.exists(cached_projected_path)
        ):
            point_picker_image = cached_projected_path
        else:
            point_picker_image = source_image_path
    else:
        print("No matching latent found. Auto-projecting the image into latent space...")
        print("This can take a while for the first run.")

        try:
            target_image = load_image_for_projection(source_image_path, g_ema.size, device)
        except Exception as e:
            print(f"Error loading existing image {source_image_path}: {e}")
            sys.exit(1)

        save_image_compat(target_image, cached_target_path)

        try:
            projected_w, projected_noises, projected_image, projection_loss = project_image_to_w(
                g_ema=g_ema,
                target_image=target_image,
                steps=args.project_steps,
                lr=args.project_lr,
                noise_regularize_weight=args.project_noise_reg,
                fullres_weight=args.project_fullres_weight,
                lowres_weight=args.project_lowres_weight,
                gradient_weight=args.project_gradient_weight,
            )
        except Exception as e:
            print(f"Error while projecting image {source_image_path}: {e}")
            sys.exit(1)

        save_image_compat(projected_image, cached_projected_path)
        torch.save(
            {
                "w": projected_w.detach().cpu(),
                "noise": [noise.detach().cpu() for noise in projected_noises],
                "source_image": source_image_path_resolved,
                "source_image_size": source_image_size,
                "source_image_mtime_ns": source_image_mtime_ns,
                "projection_loss": float(projection_loss),
                "projection_version": PROJECTION_CACHE_VERSION,
            },
            cached_latent_path,
        )

        print(f"Saved projection target image: {cached_target_path}")
        print(f"Saved projected image: {cached_projected_path}")
        print(f"Saved projected latent: {cached_latent_path}")

        if args.picker_image == "projected":
            point_picker_image = cached_projected_path
        else:
            point_picker_image = source_image_path
        w_init = projected_w
        projection_noises = [noise.detach() for noise in projected_noises]

w_init = prepare_w_latent(w_init, g_ema, device)

w_opt = w_init.detach().clone()
w_opt.requires_grad = True
optimizer = optim.Adam([w_opt], lr=0.002)

# 4. Determine Coordinates
if needs_ui_points:
    handle_points, target_points = get_points_via_ui(point_picker_image, num_points=num_points)
else:
    handle_points = [[int(y), int(x)] for y, x in cli_handle_points]
    target_points = [[int(y), int(x)] for y, x in cli_target_points]

if len(handle_points) != len(target_points) or len(handle_points) == 0:
    print("Error: At least one valid handle-target pair is required.")
    sys.exit(1)

with torch.no_grad():
    _, initial_features = g_ema(
        [w_opt.detach()],
        input_is_latent=True,
        return_features=True,
        noise=projection_noises,
        randomize_noise=False,
    )

feature_h, feature_w = initial_features.shape[2], initial_features.shape[3]
validate_points(handle_points, feature_h, feature_w, "Handle")
validate_points(target_points, feature_h, feature_w, "Target")

print(f"Starting DragGAN with {len(handle_points)} point pair(s)...")
for point_idx, (handle_point, target_point) in enumerate(zip(handle_points, target_points)):
    print(f"  Pair {point_idx + 1}: Handle {handle_point} -> Target {target_point}")

# 4. The Optimization Loop
for step in range(50): # 50 steps is usually enough for a small drag
    optimizer.zero_grad()
    
    # Notice input_is_latent=True! We are passing W, not Z.
    image, features = g_ema(
        [w_opt],
        input_is_latent=True,
        return_features=True,
        noise=projection_noises,
        randomize_noise=False,
    )
    
    # -----------------------------------------------------------------
    # Motion Supervision Loss
    # Here we calculate one motion loss per point pair and average them.
    feature_h, feature_w = features.shape[2], features.shape[3]
    point_losses = []
    original_features = []

    for pair_idx, (handle_point, target_point) in enumerate(zip(handle_points, target_points)):
        # 1. Get the Y, X coordinates of our handle and target
        hy, hx = clamp_point(handle_point[0], handle_point[1], feature_h, feature_w)
        ty, tx = clamp_point(target_point[0], target_point[1], feature_h, feature_w)
        handle_points[pair_idx] = [hy, hx]
        target_points[pair_idx] = [ty, tx]

        # 2. Extract the feature vector at the handle.
        # We use .detach() because this is our fixed target concept.
        f_original = features[:, :, hy, hx].detach()

        # 3. Calculate a 1-pixel step towards the target.
        direction_y = int(
            torch.sign(torch.tensor(float(ty - hy), device=features.device, dtype=torch.float32)).item()
        )
        direction_x = int(
            torch.sign(torch.tensor(float(tx - hx), device=features.device, dtype=torch.float32)).item()
        )

        step_y, step_x = clamp_point(hy + direction_y, hx + direction_x, feature_h, feature_w)

        # 4. Extract the features at the new 1-pixel stepped location.
        f_shifted = features[:, :, step_y, step_x]

        # 5. Motion loss for this pair.
        point_losses.append(torch.nn.functional.l1_loss(f_shifted, f_original))
        original_features.append(f_original)

    if len(point_losses) == 0:
        print("Error: No point pairs were available for optimization.")
        sys.exit(1)

    loss = torch.stack(point_losses).mean()
    
    # -----------------------------------------------------------------
    
    #if step == 0:
        #utils.save_image(image, "drag_step_00.png", normalize=True, range=(-1, 1)) #use value_range instead of range if you're using a newer version of torchvision
        # initial image before optimization starts
    
    loss.backward()
    optimizer.step()
    
    # -----------------------------------------------------------------
    # Phase B - Point Tracking
    # Here we will search the new features to find where our handle 
    # points actually moved, and update all handle coordinates.
    with torch.no_grad():
        features_norm = features / (features.norm(dim=1, keepdim=True) + 1e-8)
        updated_handle_points = []

        for f_original in original_features:
            f_original_norm = f_original / (f_original.norm(dim=1, keepdim=True) + 1e-8)

            # Compute cosine similarity across the spatial dimensions.
            similarity = torch.einsum('nc,nchw->nhw', f_original_norm, features_norm)

            # Find the location of maximum similarity for this pair.
            max_sim_idx = torch.argmax(similarity)
            new_hy, new_hx = divmod(max_sim_idx.item(), similarity.shape[2])
            updated_handle_points.append([new_hy, new_hx])

        handle_points = updated_handle_points
    
    # -----------------------------------------------------------------
    
    if step % 10 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f} | Handles: {handle_points}")
        
    if step == 49:
        save_image_compat(image, "drag_step_49.png")
        print(f"Optimization complete. Check {point_picker_image} and drag_step_49.png!")