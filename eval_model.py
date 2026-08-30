import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

val_test_transform = transforms.Compose([
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
num_classes = len(class_names)

val_data   = datasets.ImageFolder("Data_set/val", transform=val_test_transform)
test_data  = datasets.ImageFolder("Data_set/test", transform=val_test_transform)

val_loader   = DataLoader(val_data, batch_size=16, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=16, shuffle=False)

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model.load_state_dict(
    torch.load("plant_disease_efficientnet_b0.pth", map_location=device)
)
model = model.to(device)
model.eval()

def evaluate(loader, name):
    correct = 0
    total = 0
    class_correct = {cls: 0 for cls in class_names}
    class_total = {cls: 0 for cls in class_names}
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for p, l in zip(predicted, labels):
                cls_name = class_names[l.item()]
                class_total[cls_name] += 1
                if p == l:
                    class_correct[cls_name] += 1
                    
    accuracy = 100 * correct / total
    print(f"\nAccuracy on {name}: {accuracy:.2f}% ({correct}/{total})")
    print("Class-wise Accuracy:")
    for cls in class_names:
        t = class_total[cls]
        c = class_correct[cls]
        acc = (100 * c / t) if t > 0 else 0
        print(f"  {cls}: {acc:.2f}% ({c}/{t})")

evaluate(val_loader, "Validation Set")
evaluate(test_loader, "Test Set")
