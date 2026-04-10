import argparse
import os
import sys

import torch

from model import Generator


def _copy_dense_layers(dst_state, src_state, n_mlp):
    for i in range(n_mlp):
        dst_state[f"style.{i + 1}.weight"] = src_state[f"mapping.fc{i}.weight"].detach().cpu()
        dst_state[f"style.{i + 1}.bias"] = src_state[f"mapping.fc{i}.bias"].detach().cpu()


def _copy_styled_conv(dst_state, src_state, dst_prefix, src_prefix):
    dst_state[f"{dst_prefix}.conv.weight"] = src_state[f"{src_prefix}.weight"].detach().cpu().unsqueeze(0)
    dst_state[f"{dst_prefix}.conv.modulation.weight"] = src_state[f"{src_prefix}.affine.weight"].detach().cpu()
    dst_state[f"{dst_prefix}.conv.modulation.bias"] = src_state[f"{src_prefix}.affine.bias"].detach().cpu()
    dst_state[f"{dst_prefix}.noise.weight"] = src_state[f"{src_prefix}.noise_strength"].detach().cpu().reshape(1)
    dst_state[f"{dst_prefix}.activate.bias"] = src_state[f"{src_prefix}.bias"].detach().cpu()


def _copy_torgb(dst_state, src_state, dst_prefix, src_prefix):
    dst_state[f"{dst_prefix}.conv.weight"] = src_state[f"{src_prefix}.weight"].detach().cpu().unsqueeze(0)
    dst_state[f"{dst_prefix}.conv.modulation.weight"] = src_state[f"{src_prefix}.affine.weight"].detach().cpu()
    dst_state[f"{dst_prefix}.conv.modulation.bias"] = src_state[f"{src_prefix}.affine.bias"].detach().cpu()
    dst_state[f"{dst_prefix}.bias"] = src_state[f"{src_prefix}.bias"].detach().cpu().reshape(1, 3, 1, 1)


def _copy_noises(dst_state, src_state, size):
    dst_state["noises.noise_0"] = src_state["synthesis.b4.conv1.noise_const"].detach().cpu().unsqueeze(0).unsqueeze(0)

    conv_idx = 0
    res = 8
    while res <= size:
        dst_state[f"noises.noise_{conv_idx + 1}"] = (
            src_state[f"synthesis.b{res}.conv0.noise_const"].detach().cpu().unsqueeze(0).unsqueeze(0)
        )
        dst_state[f"noises.noise_{conv_idx + 2}"] = (
            src_state[f"synthesis.b{res}.conv1.noise_const"].detach().cpu().unsqueeze(0).unsqueeze(0)
        )
        conv_idx += 2
        res *= 2


def convert(network_pkl, out_path, ada_repo, channel_multiplier=2):
    sys.path.append(ada_repo)

    import dnnlib
    import legacy

    with dnnlib.util.open_url(network_pkl) as f:
        network = legacy.load_network_pkl(f)

    G_ema = network["G_ema"].eval().requires_grad_(False)
    src = G_ema.state_dict()

    size = int(G_ema.img_resolution)
    n_mlp = int(G_ema.mapping.num_layers)

    g = Generator(size, 512, n_mlp, channel_multiplier=channel_multiplier)
    dst = g.state_dict()

    _copy_dense_layers(dst, src, n_mlp)

    dst["input.input"] = src["synthesis.b4.const"].detach().cpu().unsqueeze(0)
    _copy_styled_conv(dst, src, "conv1", "synthesis.b4.conv1")
    _copy_torgb(dst, src, "to_rgb1", "synthesis.b4.torgb")

    i = 0
    res = 8
    while res <= size:
        _copy_styled_conv(dst, src, f"convs.{i}", f"synthesis.b{res}.conv0")
        _copy_styled_conv(dst, src, f"convs.{i + 1}", f"synthesis.b{res}.conv1")
        _copy_torgb(dst, src, f"to_rgbs.{i // 2}", f"synthesis.b{res}.torgb")
        i += 2
        res *= 2

    _copy_noises(dst, src, size)

    g.load_state_dict(dst, strict=True)

    ckpt = {
        "g_ema": dst,
        "latent_avg": src["mapping.w_avg"].detach().cpu(),
    }
    torch.save(ckpt, out_path)


def main():
    parser = argparse.ArgumentParser(description="Convert StyleGAN2-ADA .pkl to rosinality .pt")
    parser.add_argument("network_pkl", type=str, help="Path to StyleGAN2-ADA .pkl")
    parser.add_argument("--out", type=str, default=None, help="Output .pt path")
    parser.add_argument(
        "--ada_repo",
        type=str,
        default="/home/project/MotionGAN/stylegan2-ada-pytorch",
        help="Path to stylegan2-ada-pytorch repository",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier (FFHQ config-f uses 2)",
    )
    args = parser.parse_args()

    out_path = args.out
    if out_path is None:
        name = os.path.splitext(os.path.basename(args.network_pkl))[0]
        out_path = name + ".pt"

    convert(
        network_pkl=args.network_pkl,
        out_path=out_path,
        ada_repo=args.ada_repo,
        channel_multiplier=args.channel_multiplier,
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
