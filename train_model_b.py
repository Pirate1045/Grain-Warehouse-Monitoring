import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 55)
print("  FILE 4 of 8 — train_model_b.py")
print("  Training ResNet-50 insect classifier")
print("=" * 55)

os.makedirs('models',  exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ── Check class_names.json exists ─────────────────────────────────────
if not os.path.exists('models/class_names.json'):
    print("\nERROR: models/class_names.json not found.")
    print("Run preprocess_images.py first.")
    sys.exit(1)

with open('models/class_names.json') as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)
print(f"\nClasses ({NUM_CLASSES}): {class_names}")

# ── Image directory ───────────────────────────────────────────────────
IMAGE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data', 'insect_images'
)

# ── Transforms ───────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomAffine(degrees=0, shear=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── Load dataset ──────────────────────────────────────────────────────
full = datasets.ImageFolder(IMAGE_DIR, transform=train_tf)
n_train = int(0.7 * len(full))
n_val   = len(full) - n_train

train_ds, val_ds = random_split(
    full, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)
val_ds.dataset.transform = val_tf

train_loader = DataLoader(train_ds, batch_size=16,
                          shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=16,
                          shuffle=False, num_workers=0)

print(f"Train : {len(train_ds)}  Val : {len(val_ds)}")

# ── Load ResNet-50 ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Freeze first 7 child modules (conv1 through layer2)
for i, child in enumerate(model.children()):
    if i < 7:
        for param in child.parameters():
            param.requires_grad = False

# Replace final fully connected layer
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model    = model.to(device)

# ── Optimiser ─────────────────────────────────────────────────────────
opt = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0003, momentum=0.9
)
loss_fn   = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(
    opt, step_size=7, gamma=0.1
)

# ── Training loop ─────────────────────────────────────────────────────
best_acc       = 0.0
patience_count = 0
PATIENCE       = 5
MAX_EPOCHS     = 20
train_accs     = []
val_accs       = []

print("\nTraining started...")
print(f"  Max epochs : {MAX_EPOCHS}")
print(f"  Batch size : 16")
print(f"  LR         : 0.0003\n")

for epoch in range(1, MAX_EPOCHS + 1):
    # Train
    model.train()
    running_loss = 0.0
    correct = total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad()
        out  = model(imgs)
        loss = loss_fn(out, labels)
        loss.backward()
        opt.step()
        running_loss += loss.item()
        correct      += (out.argmax(1) == labels).sum().item()
        total        += labels.size(0)
    train_acc = correct / total * 100
    train_accs.append(train_acc)
    scheduler.step()

    # Validate
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    val_acc = correct / total * 100
    val_accs.append(val_acc)

    print(f"  Epoch {epoch:2d}/{MAX_EPOCHS}  "
          f"loss={running_loss/len(train_loader):.4f}  "
          f"train={train_acc:.1f}%  val={val_acc:.1f}%")

    if val_acc > best_acc:
        best_acc       = val_acc
        patience_count = 0
        torch.save(model.state_dict(),
                   'models/resnet50_insects.pt')
        print(f"    → Saved best model ({val_acc:.1f}%)")
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

print(f"\nBest validation accuracy: {best_acc:.2f}%")

# ── Full evaluation ───────────────────────────────────────────────────
model.load_state_dict(
    torch.load('models/resnet50_insects.pt', map_location='cpu')
)
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        preds = model(imgs.to(device)).argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_true.extend(labels.tolist())

print("\nClassification Report:")
print(classification_report(
    all_true, all_preds, target_names=class_names
))

# ── Confusion matrix ──────────────────────────────────────────────────
cm = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('ResNet-50 Insect Classifier\nConfusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('outputs/resnet_confusion_matrix.png', dpi=150)
plt.close()

# ── Accuracy curve ────────────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(train_accs, label='Train Accuracy', color='steelblue')
plt.plot(val_accs,   label='Val Accuracy',   color='darkorange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('ResNet-50 Training Curve')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/resnet_accuracy_curve.png', dpi=150)
plt.close()

print("\n  Model saved:  models/resnet50_insects.pt")
print("  Charts saved: outputs/resnet_confusion_matrix.png")
print("                outputs/resnet_accuracy_curve.png")
print("\n✓ DONE — train_model_b.py complete")
print("  Next: python gradcam.py")
