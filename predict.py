import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ===============================
# 1. IMAGE PATH (YOUR IMAGE)
# ===============================
image_path = r"C:\Users\DELL\OneDrive\Documents\Pictures\images (1).jpg"

# ===============================
# 2. DEVICE
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# 3. IMAGE TRANSFORM
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ===============================
# 4. LOAD IMAGE
# ===============================
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# ===============================
# 5. CLASS NAMES (EXACT FROM TRAINING)
# ===============================
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

# ===============================
# 6. LOAD TRAINED MODEL
# ===============================
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(class_names)
)

model.load_state_dict(
    torch.load("plant_disease_efficientnet_b0.pth", map_location=device)
)

model = model.to(device)
model.eval()

# ===============================
# 7. PREDICTION
# ===============================
with torch.no_grad():
    outputs = model(image)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted_index = torch.max(probabilities, 1)

predicted_class = class_names[predicted_index.item()]
confidence = confidence.item() * 100

# ===============================
# 8. TREATMENT SUGGESTIONS
# ===============================
treatment = {
    "Pepper__bell___Bacterial_spot": "Use disease-free seeds and apply recommended bactericide.",
    "Pepper__bell___healthy": "Plant is healthy. Maintain proper watering and nutrition.",
    "Potato___Early_blight": "Remove infected leaves and apply fungicide.",
    "Potato___Late_blight": "Apply fungicide immediately and avoid excess moisture.",
    "Potato___healthy": "Plant is healthy. Continue good agricultural practices.",
    "Tomato_Early_blight": "Remove infected leaves and spray copper-based fungicide.",
    "Tomato_Late_blight": "Use fungicide and reduce humidity around plants.",
    "Tomato_healthy": "Plant is healthy. Maintain soil nutrients and watering."
}

# ===============================
# 9. OUTPUT
# ===============================
print("\n🌱 PLANT DISEASE DETECTION RESULT")
print("--------------------------------")
print(f"Disease Detected : {predicted_class}")
print(f"Confidence       : {confidence:.2f}%")
print(f"Suggested Action : {treatment[predicted_class]}")
