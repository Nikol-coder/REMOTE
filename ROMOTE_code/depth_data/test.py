import os
import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
from tqdm import tqdm  

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# init
encoder = 'vitl'  
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

input_dir = "UMKE_IMG"
output_dir = "depth_data_umke"
os.makedirs(output_dir, exist_ok=True)

for filename in tqdm(os.listdir(input_dir), desc="Processing images"):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  
        
        img_path = os.path.join(input_dir, filename)
        raw_img = cv2.imread(img_path)
        
        if raw_img is None:
            print(f"Failed to read {img_path}")
            continue
            
        depth_map = model.infer_image(raw_img)  
        
        base_name = os.path.splitext(filename)[0]  
        npz_path = os.path.join(output_dir, f"{base_name}.npz")
        
        np.savez(npz_path, depth_map=depth_map)  
        print(f"Saved depth map: {npz_path}")