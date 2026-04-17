
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument(
    "--image",
    type=str,
    default="drag_step_00.png",
    help="Image to click on.",
)
parser.add_argument(
    "--num-points",
    type=int,
    default=0,
    help="Number of handle-target pairs to collect. Use 0 for manual finish mode.",
)
args = parser.parse_args()

if args.num_points < 0:
    raise SystemExit("Error: --num-points must be 0 or greater")

print(f"Loading image: {args.image}")
img = mpimg.imread(args.image)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img)

if args.num_points == 0:
    ax.set_title("Manual mode: click Handle then Target pairs, press Enter when done")
else:
    ax.set_title(
        f"Pick {args.num_points} pair(s): Handle then Target for each pair"
    )

print("Waiting for your clicks in the pop-up window...")
if args.num_points == 0:
    coords = plt.ginput(-1, timeout=-1)
else:
    coords = plt.ginput(2 * args.num_points, timeout=-1)

if len(coords) < 2:
    print("You must click at least one handle-target pair. Try again.")
    plt.close()
    raise SystemExit(1)

if len(coords) % 2 != 0:
    print("You must click an even number of points (handle/target pairs). Try again.")
    plt.close()
    raise SystemExit(1)

if args.num_points != 0 and len(coords) != 2 * args.num_points:
    print(f"You must click exactly {2 * args.num_points} points. Try again.")
    plt.close()
    raise SystemExit(1)

num_pairs = len(coords) // 2

handles = []
targets = []
for point_idx in range(num_pairs):
    # Matplotlib returns (X, Y) on the 1024x1024 image.
    handle_x, handle_y = coords[2 * point_idx]
    target_x, target_y = coords[2 * point_idx + 1]

    # 1. Scale down to the 256x256 feature map (divide by 4).
    # 2. Swap from (X, Y) to PyTorch's [Y, X] format.
    hy, hx = int(handle_y / 4), int(handle_x / 4)
    ty, tx = int(target_y / 4), int(target_x / 4)

    handles.append([hy, hx])
    targets.append([ty, tx])

print("\n=== Success! Use these CLI arguments with drag_optimize.py ===")
for handle in handles:
    print(f"--handle {handle[0]} {handle[1]}")
for target in targets:
    print(f"--target {target[0]} {target[1]}")

plt.close()