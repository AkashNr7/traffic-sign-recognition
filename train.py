import os
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# === Hyperparameters ===
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
IMG_SIZE = (32, 32)  

# === Paths ===
CSV_PATH = "archive/Train.csv"
IMAGE_ROOT = "archive"  
MODEL_PATH = "sign_model.pth"

# === Custom Dataset ===
class TrafficSignDataset(Dataset):
    def __init__(self, csv_path, image_root, transform=None):
        self.data = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_root, row['Path'])
        label = int(row['ClassId'])

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label

# === Image Preprocessing ===
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# === Load Dataset ===
dataset = TrafficSignDataset(CSV_PATH, IMAGE_ROOT, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === CNN Model normalization and to 32X32 size ====

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # [B, 32, 16, 16]
        x = self.pool(torch.relu(self.conv2(x)))  # [B, 64, 8, 8]
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# === Training ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(num_classes=43).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(" Starting Training...")
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f" Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

# === Save Model ===
torch.save(model.state_dict(), MODEL_PATH)
print(f" Model saved as {MODEL_PATH}")



