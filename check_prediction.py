import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

image_path = r"C:\Users\DELL\OneDrive\Documents\Pictures\leaves_test\f1.jpg"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy"
]

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
model.load_state_dict(torch.load("plant_disease_efficientnet_b0.pth", map_location=device))
model = model.to(device)
model.eval()

if os.path.exists(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
    
    print("Class Probabilities:")
    for name, prob in zip(class_names, probabilities):
        print(f"  {name}: {prob.item() * 100:.4f}%")
else:
    print(f"File not found: {image_path}")
