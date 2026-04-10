import torch
from torch import optim
from model import Generator
from torchvision import utils

device = 'cuda'
# g_ema should throw error because pylance checks for static types, but it will work when you run the script
# 1. Initialize and load the model (This silences Pylance)
g_ema = Generator(1024, 512, 8).to(device)
checkpoint = torch.load('ffhq.pt') # Make sure this path is correct
g_ema.load_state_dict(checkpoint['g_ema'])

g_ema.eval() # Generator weights remain frozen!

# 1. Generate an initial W latent code (instead of Z)
sample_z = torch.randn(1, 512, device=device)
with torch.no_grad():
    # Pass Z through the mapping network to get W
    w_init = g_ema.style(sample_z) 

# 2. Make our latent code a trainable parameter
w_opt = w_init.detach().clone()
w_opt.requires_grad = True

# 3. Setup the Optimizer
# DragGAN typically uses Adam. A learning rate around 2e-3 is a good starting point.
optimizer = optim.Adam([w_opt], lr=0.002)

# Define a fake Handle and Target point (y, x coordinates at the 256x256 resolution)
handle_point = [120, 150] 
target_point = [120, 170] 

print("Starting DragGAN Optimization Loop...")

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
    
    if step == 0:
        utils.save_image(image, "drag_step_00.png", normalize=True, range=(-1, 1)) #use value_range instead of range if you're using a newer version of torchvision
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