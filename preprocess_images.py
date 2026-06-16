import os
import sys
import json
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

print("=" * 55)
print("  FILE 2 of 8 — preprocess_images.py")
print("  Loading and preparing insect images")
print("=" * 55)

# ── Create folders ────────────────────────────────────────────────────
os.makedirs('models',  exist_ok=True)
os.makedirs('outputs', exist_ok=True)

IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'data', 'insect_images')

# ── Check folders exist and have images ──────────────────────────────
if not os.path.exists(IMAGE_DIR):
    print(f"\nERROR: Folder not found: {IMAGE_DIR}")
    print("Create these folders and add insect photos:")
    for sp in ['LGB', 'RGB', 'RFB', 'RW', 'ST']:
        print(f"  data/insect_images/{sp}/")
    sys.exit(1)

species_folders = [f for f in os.listdir(IMAGE_DIR)
                   if os.path.isdir(os.path.join(IMAGE_DIR, f))]
print(f"\nSpecies folders found: {species_folders}")

total_images = 0
for sp in species_folders:
    sp_path = os.path.join(IMAGE_DIR, sp)
    imgs = [f for f in os.listdir(sp_path)
            if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]
    print(f"  {sp}: {len(imgs)} images")
    total_images += len(imgs)

if total_images == 0:
    print("\nERROR: No images found in any species folder.")
    print("Add at least 20 images per species folder.")
    sys.exit(1)

print(f"\nTotal images: {total_images}")

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
full_dataset = datasets.ImageFolder(IMAGE_DIR, transform=train_tf)
print(f"\nClasses : {full_dataset.classes}")
print(f"Total   : {len(full_dataset)}")

# ── Split 70 / 30 ────────────────────────────────────────────────────
n_train = int(0.7 * len(full_dataset))
n_val   = len(full_dataset) - n_train

train_ds, val_ds = random_split(
    full_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)
val_ds.dataset.transform = val_tf

# ── DataLoaders ───────────────────────────────────────────────────────
train_loader = DataLoader(train_ds, batch_size=16,
                          shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=16,
                          shuffle=False, num_workers=0)

print(f"Train   : {len(train_ds)}")
print(f"Val     : {len(val_ds)}")

# ── Save class names ──────────────────────────────────────────────────
with open('models/class_names.json', 'w') as f:
    json.dump(full_dataset.classes, f)

print("\n✓ DONE — preprocess_images.py complete")
print("  class_names.json saved to models/")
print("  Next: python train_model_a.py")
