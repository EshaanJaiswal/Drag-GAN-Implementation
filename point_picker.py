

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print("Loading image...")
img = mpimg.imread('drag_step_00.png')

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img)
ax.set_title("Click 1: Handle (What to move) | Click 2: Target (Where to go)")

print("Waiting for your clicks in the pop-up window...")

#we capture exactly 2 clicks using ginput...

coords = plt.ginput(2, timeout=-1) #we are waiting indefinitely

if len(coords) == 2:
    # Matplotlib returns (X, Y) on the 1024x1024 image
    handle_x, handle_y = coords[0]
    target_x, target_y = coords[1]

    # 1. Scale down to the 256x256 feature map (divide by 4)
    # 2. Swap from (X, Y) to PyTorch's [Y, X] format
    hy, hx = int(handle_y / 4), int(handle_x / 4)
    ty, tx = int(target_y / 4), int(target_x / 4)

    print("\n=== Success! Copy these into drag_optimize.py ===")
    print(f"handle_point = [{hy}, {hx}]")
    print(f"target_point = [{ty}, {tx}]")
else:
    print("You didn't click twice. Try again.")
    
plt.close()