"""
debug_single_image.py
----------------------
Run this with an image path to test the EXACT same code path app.py uses,
but with no Flask, no browser, no frontend involved. This isolates whether
the bug is in the model/transform logic itself.

Usage:
    python debug_single_image.py "path/to/your/image.jpg"
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from PIL import Image, ImageOps

if len(sys.argv) < 2:
    print("Usage: python debug_single_image.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

MODEL_PATH = "plant_disease_efficientnet_b0.pth"
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy",
]

# --- Sanity check #1: does CLASS_NAMES match what ImageFolder actually sees? ---
try:
    train_data = datasets.ImageFolder("Data_set/train")
    actual_order = train_data.classes
    print("=" * 60)
    print("CLASS ORDER CHECK")
    print("=" * 60)
    print("CLASS_NAMES in app.py :", CLASS_NAMES)
    print("ImageFolder sees      :", actual_order)
    if actual_order == CLASS_NAMES:
        print(">>> MATCH - class order is correct.\n")
    else:
        print(">>> MISMATCH!! This is very likely your bug. Fix CLASS_NAMES")
        print(">>> in app.py to exactly match the ImageFolder order above.\n")
except Exception as e:
    print(f"(Skipped class order check - couldn't find Data_set/train here: {e})\n")

# --- Load model exactly like app.py does ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(in_features, len(CLASS_NAMES)),
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- Load image exactly like app.py does ---
image = Image.open(image_path)
image = ImageOps.exif_transpose(image)
image = image.convert("RGB")
print("=" * 60)
print("IMAGE INFO")
print("=" * 60)
print("Path  :", image_path)
print("Size  :", image.size)
print("Mode  :", image.mode)

tensor = transform(image).unsqueeze(0).to(device)
print("Tensor shape :", tensor.shape)
print("Tensor min/max/mean :", tensor.min().item(), tensor.max().item(), tensor.mean().item())

with torch.no_grad():
    logits = model(tensor)
    probs = F.softmax(logits, dim=1)[0]

all_probs = {CLASS_NAMES[i]: round(probs[i].item() * 100, 2) for i in range(len(CLASS_NAMES))}
top3 = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]

print("\n" + "=" * 60)
print("PREDICTION")
print("=" * 60)
for name, p in top3:
    print(f"  {name}: {p}%")
