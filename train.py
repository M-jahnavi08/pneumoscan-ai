# train.py
# We use EfficientNetB3 — a powerful pretrained model — and fine-tune it
# on our X-ray images. This is called TRANSFER LEARNING.

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ── 0. Setup ────────────────────────────────────────────────────────────────
# Check if we have a GPU (much faster). If not, use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

DATA_DIR   = "chest_xray"
BATCH_SIZE = 32
EPOCHS     = 3
IMG_SIZE   = 224
LR         = 1e-4  # learning rate

# ── 1. Data Transforms (preprocessing) ─────────────────────────────────────
# Think of this as "preparing" images before feeding to the model.
# Training images get random flips/rotations so the model learns to generalise.

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),           # randomly mirror image
    transforms.RandomRotation(10),               # randomly rotate ±10°
    transforms.ColorJitter(brightness=0.2,
                           contrast=0.2),        # slight brightness changes
    transforms.ToTensor(),                       # convert image to numbers
    transforms.Normalize([0.485, 0.456, 0.406], # normalise (standard values
                         [0.229, 0.224, 0.225])  #   for pretrained models)
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── 2. Load datasets ─────────────────────────────────────────────────────────
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"),
                                     transform=train_transforms)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),
                                     transform=val_test_transforms)
test_dataset  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),
                                     transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

print(f"📊 Classes: {train_dataset.classes}")
print(f"📦 Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# ── 3. Handle Class Imbalance ────────────────────────────────────────────────
# There are more PNEUMONIA images. We give more "weight" to NORMAL
# so the model doesn't just predict PNEUMONIA for everything.
class_counts  = np.array([len(os.listdir(os.path.join(DATA_DIR, "train", c)))
                           for c in train_dataset.classes])
class_weights = 1.0 / class_counts
class_weights = torch.tensor(class_weights / class_weights.sum(),
                              dtype=torch.float).to(device)
print(f"⚖️  Class weights: {class_weights}")  

# ── 4. Build the Model ───────────────────────────────────────────────────────
# EfficientNet was trained on 1 million images. We take its "brain" (features)
# and just replace the last layer so it outputs 2 classes (Normal / Pneumonia).

model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

# Freeze early layers — keep their pretrained knowledge
for param in model.parameters():
    param.requires_grad = False

# Only train the final classifier layer
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_features, 2)   # 2 output classes
)

model = model.to(device)
print("✅ Model built!")

# ── 5. Loss & Optimiser ──────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.classifier.parameters(), lr=LR)

# Learning rate scheduler: reduces LR when improvement stalls
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                  patience=2, factor=0.5)

# ── 6. Training Loop ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs    = model(images)
            loss       = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# ── 7. Run Training ──────────────────────────────────────────────────────────
best_val_acc  = 0
best_model_wts = copy.deepcopy(model.state_dict())
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

print("\n🚀 Starting training...\n")
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader,
                                            criterion, optimizer)
    val_loss,   val_acc   = evaluate(model, val_loader, criterion)
    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc   = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), "best_model.pth")
        print(f"   💾 Best model saved! (val_acc={best_val_acc:.4f})")

print(f"\n✅ Training complete. Best Val Accuracy: {best_val_acc:.4f}")

# ── 8. Plot Training Curves ──────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history["train_loss"], label="Train Loss", marker='o')
ax1.plot(history["val_loss"],   label="Val Loss",   marker='o')
ax1.set_title("Loss over Epochs"); ax1.set_xlabel("Epoch")
ax1.legend(); ax1.grid(True)

ax2.plot(history["train_acc"], label="Train Acc", marker='o')
ax2.plot(history["val_acc"],   label="Val Acc",   marker='o')
ax2.set_title("Accuracy over Epochs"); ax2.set_xlabel("Epoch")
ax2.legend(); ax2.grid(True)

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()
print("📈 Training curves saved!")

# ── 9. Evaluate on Test Set ──────────────────────────────────────────────────
model.load_state_dict(best_model_wts)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        preds  = model(images).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print("\n📋 Classification Report:")
print(classification_report(all_labels, all_preds,
                             target_names=train_dataset.classes))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=train_dataset.classes,
            yticklabels=train_dataset.classes)
plt.title("Confusion Matrix"); plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.savefig("confusion_matrix.png")
plt.show()
print("📊 Confusion matrix saved!")