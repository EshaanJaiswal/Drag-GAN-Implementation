import torch
from model import Generator

# Load your model (adjust resolution and path)
device = 'cuda'
g_ema = Generator(1024, 512, 8).to(device)
checkpoint = torch.load('ffhq.pt')
g_ema.load_state_dict(checkpoint['g_ema'])
g_ema.eval()

# Generate random noise
sample_z = torch.randn(1, 512, device=device)

# Forward pass with feature extraction
with torch.no_grad():
    image, features = g_ema([sample_z], return_features=True)

print(f"Final Image Shape: {image.shape}")
print(f"Extracted Features Shape: {features.shape}") 
# Expected feature shape: [1, Channels, 256, 256]