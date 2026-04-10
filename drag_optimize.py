import torch
from torch import optim
from model import Generator
from torchvision import utils
import argparse
import sys
import os

# 1. Interactive UI Function
def get_points_via_ui(image_path='drag_step_00.png'):
    print("No coordinates provided via CLI. Launching interactive picker...")
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        img = mpimg.imread(image_path)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)
        ax.set_title("Click 1: Handle | Click 2: Target (Close window when done)")
        coords = plt.ginput(2, timeout=-1)
        plt.close()
        
        if len(coords) == 2:
            hy, hx = int(coords[0][1] / 4), int(coords[0][0] / 4)
            ty, tx = int(coords[1][1] / 4), int(coords[1][0] / 4)
            return [hy, hx], [ty, tx]
        else:
            print("Error: You didn't click twice. Exiting.")
            sys.exit(1)
    except Exception as e:
        print(f"\n[!] UI Failed: {e}")
        print("[!] Run with arguments instead: python3 drag_optimize.py --handle Y X --target Y X\n")
        sys.exit(1)

# 2. Setup Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--handle", type=int, nargs=2, default=None)
parser.add_argument("--target", type=int, nargs=2, default=None)
args = parser.parse_args()

device = 'cuda'
torch.manual_seed(42) # Ensure we generate the SAME face every run!

g_ema = Generator(1024, 512, 8).to(device)
checkpoint = torch.load('ffhq.pt') 
g_ema.load_state_dict(checkpoint['g_ema'])
g_ema.eval() 

sample_z = torch.randn(1, 512, device=device)
with torch.no_grad():
    w_init = g_ema.style(sample_z) 

w_opt = w_init.detach().clone()
w_opt.requires_grad = True
optimizer = optim.Adam([w_opt], lr=0.002)

# 3. Generate and save the initial image BEFORE picking points
with torch.no_grad():
    initial_image, _ = g_ema([w_opt], input_is_latent=True, return_features=True, randomize_noise=False)
    utils.save_image(initial_image, "drag_step_00.png", normalize=True, range=(-1, 1))

# 4. Determine Coordinates
if args.handle is None or args.target is None:
    handle_point, target_point = get_points_via_ui('drag_step_00.png')
else:
    handle_point = args.handle
    target_point = args.target

print(f"Starting DragGAN... Moving Handle {handle_point} to Target {target_point}")

# 4. The Optimization Loop
for step in range(50): # 50 steps is usually enough for a small drag
    optimizer.zero_grad()
    
    # Notice input_is_latent=True! We are passing W, not Z.
    image, features = g_ema([w_opt], input_is_latent=True, return_features=True, randomize_noise=False) 
    
    # -----------------------------------------------------------------
    # Motion Supervision Loss
    # Here we will calculate the loss that forces the feature patch at 
    # 'handle_point' to move towards 'target_point'.
    
    # 1. Get the Y, X coordinates of our handle and target
    hy, hx = int(handle_point[0]), int(handle_point[1])
    ty, tx = int(target_point[0]), int(target_point[1])
    
    # 2. Extract the 128-dim feature vector at the handle.
    # We use .detach() because we want this to be our fixed target concept.
    # We do NOT want gradients flowing backwards through our ground truth!
    f_original = features[:, :, hy, hx].detach()
    
    # 3. Calculate a 1-pixel step towards the target
    # torch.sign() turns distances into -1, 0, or 1 to give us a clean step direction
    direction_y = torch.sign(torch.tensor(ty - hy, dtype=torch.float32)).long()
    direction_x = torch.sign(torch.tensor(tx - hx, dtype=torch.float32)).long()
    
    step_y = hy + direction_y
    step_x = hx + direction_x
    
    # 4. Extract the features at the NEW 1-pixel stepped location
    f_shifted = features[:, :, step_y, step_x]
    
    # 5. The Motion Loss: Force the new location to match our original concept
    loss = torch.nn.functional.l1_loss(f_shifted, f_original)
    
    # -----------------------------------------------------------------
    
    #if step == 0:
        #utils.save_image(image, "drag_step_00.png", normalize=True, range=(-1, 1)) #use value_range instead of range if you're using a newer version of torchvision
        # initial image before optimization starts
    
    loss.backward()
    optimizer.step()
    
    # -----------------------------------------------------------------
    # Phase B - Point Tracking
    # Here we will search the new features to find where our handle 
    # point actually moved, and update 'handle_point' coordinates.
    with torch.no_grad():
        # Calculate the cosine similarity between the original feature and all features
        f_original_norm = f_original / (f_original.norm(dim=1, keepdim=True) + 1e-8)
        features_norm = features / (features.norm(dim=1, keepdim=True) + 1e-8)
        
        # Compute cosine similarity across the spatial dimensions
        similarity = torch.einsum('nc,nchw->nhw', f_original_norm, features_norm)
        
        # Find the location of the maximum similarity
        max_sim_idx = torch.argmax(similarity)
        new_hy, new_hx = divmod(max_sim_idx.item(), similarity.shape[2])
        
        # Update handle_point to the new location
        handle_point = [new_hy, new_hx] 
    
    # -----------------------------------------------------------------
    
    if step % 10 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f} | Handle moving to: [{step_y}, {step_x}]")
        
    if step == 49:
        utils.save_image(image, "drag_step_49.png", normalize=True, range=(-1, 1))
        print("Optimization complete. Check drag_step_00.png and drag_step_49.png!")