import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from collections import Counter
from PIL import Image

# CONFIG
DATA_DIR = r"C:\Users\Yaven_Singh\Desktop\My Eye Tracking Model\My Eye Tracking Model\Eye Tracking Dataset\Images"
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# DATASET
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# MODEL
weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# TRAINING
# Count class distribution
labels = [label for _, label in dataset.samples]
class_counts = Counter(labels)
class_weights = 1. / torch.tensor([class_counts[0], class_counts[1]], dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"[Epoch {epoch+1}] Loss: {running_loss/len(train_loader):.4f}")
    scheduler.step()

# EVALUATION
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nConfusion Matrix:\n", confusion_matrix(all_labels, all_preds))
print("\nClassification Report:\n", classification_report(all_labels, all_preds))
print("✅ Done.")

def predict_image(image_path, model):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    
    # Apply the same transforms used during training
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        _, prediction = torch.max(output, 1)
    
    class_names = dataset.classes
    predicted_label = class_names[prediction.item()]
    print(f"🧠 Predicted Class: {predicted_label}")
    return predicted_label

# 🔍 Test with a new scanpath image
scanpath_img = r"C:\Users\Yaven_Singh\Desktop\My Eye Tracking Model\My Eye Tracking Model\scanpath.png"
predict_image(scanpath_img, model)
