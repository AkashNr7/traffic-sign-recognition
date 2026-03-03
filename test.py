import os
import pandas as pd
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# === Configs ===
CSV_PATH = "archive/Test.csv"
IMAGE_ROOT = "archive"
MODEL_PATH = "sign_model.pth"
BATCH_SIZE = 64
IMG_SIZE = (32, 32)

# === Preprocessing ===
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# === Dataset Class ===
class TrafficSignTestDataset(Dataset):
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
            raise FileNotFoundError(f"Missing: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label

# === Model Definition ===
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

# === Load Test Data ===
dataset = TrafficSignTestDataset(CSV_PATH, IMAGE_ROOT, transform=transform)
test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(num_classes=43).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# === Evaluation ===
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f" Test Accuracy: {accuracy:.2f}%")


