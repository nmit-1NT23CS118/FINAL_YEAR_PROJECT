import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ===============================
# 1. DEVICE SETUP
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# 2. IMAGE TRANSFORMS
# ===============================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ===============================
# 3. LOAD DATASETS
# ===============================
train_data = datasets.ImageFolder("Data_set/train", transform=train_transform)
val_data   = datasets.ImageFolder("Data_set/val", transform=val_test_transform)
test_data  = datasets.ImageFolder("Data_set/test", transform=val_test_transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=16, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=16, shuffle=False)

class_names = train_data.classes
num_classes = len(class_names)

print("Classes:", class_names)
print("Train images:", len(train_data))
print("Validation images:", len(val_data))
print("Test images:", len(test_data))

# ===============================
# 4. LOAD EFFICIENTNET-B0
# ===============================
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Replace classifier layer
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model = model.to(device)

# ===============================
# 5. LOSS & OPTIMIZER
# ===============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ===============================
# 6. TRAINING WITH VALIDATION
# ===============================
import sys
epochs = 2
if len(sys.argv) > 1:
    try:
        epochs = int(sys.argv[1])
        print(f"Overriding epochs to {epochs}")
    except ValueError:
        pass

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    print("-" * 30)

    # ---- TRAIN ----
    model.train()
    train_loss = 0
    total_batches = len(train_loader)

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
            print(f"  Batch {batch_idx + 1}/{total_batches} - Loss: {loss.item():.4f}")

    avg_train_loss = train_loss / total_batches
    print(f"Training Loss: {avg_train_loss:.4f}")

    # ---- VALIDATION ----
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

# ===============================
# 7. FINAL TEST ACCURACY
# ===============================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print("\nFinal Test Accuracy:", f"{test_accuracy:.2f}%")

# ===============================
# 8. SAVE MODEL
# ===============================
torch.save(model.state_dict(), "plant_disease_efficientnet_b0.pth")
print("✅ Model saved as plant_disease_efficientnet_b0.pth")
