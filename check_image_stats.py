import os
from PIL import Image

image_path = r"C:\Users\DELL\OneDrive\Documents\Pictures\leaves_test\f1.jpg"
if os.path.exists(image_path):
    img = Image.open(image_path)
    print(f"Format: {img.format}")
    print(f"Size: {img.size}")
    print(f"Mode: {img.mode}")
    # Print some pixel stats
    import numpy as np
    img_np = np.array(img)
    print(f"Mean R, G, B: {img_np.mean(axis=(0,1))}")
    print(f"Std R, G, B: {img_np.std(axis=(0,1))}")
else:
    print("Image not found")
