import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Use subset of data for quick benchmarking
train_data = datasets.ImageFolder("Data_set/train", transform=train_transform)
# Take 160 images (10 batches)
subset_indices = list(range(160))
subset_data = torch.utils.data.Subset(train_data, subset_indices)
train_loader = DataLoader(subset_data, batch_size=16, shuffle=True)

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

start_time = time.time()
model.train()
for i, (images, labels) in enumerate(train_loader):
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f"Batch {i+1} completed")

elapsed = time.time() - start_time
print(f"Time for 10 batches (160 images): {elapsed:.2f} seconds")
print(f"Estimated time for full train set (6390 images): {elapsed * (6390/160) / 60:.2f} minutes")
