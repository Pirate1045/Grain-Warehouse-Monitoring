import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm

print("=" * 55)
print("  FILE 5 of 8 — gradcam.py")
print("  Generating Grad-CAM heat maps for each species")
print("=" * 55)

os.makedirs('outputs/gradcam', exist_ok=True)

# ── Check required files ──────────────────────────────────────────────
for f in ['models/resnet50_insects.pt', 'models/class_names.json']:
    if not os.path.exists(f):
        print(f"\nERROR: {f} not found.")
        print("Run train_model_b.py first.")
        sys.exit(1)

with open('models/class_names.json') as f:
    class_names = json.load(f)

# ── Load model ────────────────────────────────────────────────────────
device = torch.device('cpu')
model  = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(
    torch.load('models/resnet50_insects.pt', map_location='cpu')
)
model.eval()
print(f"\nModel loaded. Classes: {class_names}")

# ── Grad-CAM hooks ────────────────────────────────────────────────────
gradients  = []
activations= []

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0].detach())

def forward_hook(module, input, output):
    activations.append(output.detach())

target_layer = model.layer4[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# ── Image transform ───────────────────────────────────────────────────
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def run_gradcam(image_path, save_path):
    gradients.clear()
    activations.clear()

    # Load and preprocess
    img_pil = Image.open(image_path).convert('RGB')
    inp     = tf(img_pil).unsqueeze(0)
    inp.requires_grad_()

    # Forward pass
    out      = model(inp)
    pred_idx = out.argmax(dim=1).item()
    pred_cls = class_names[pred_idx]
    prob     = F.softmax(out, dim=1)[0, pred_idx].item()

    # Backward pass
    model.zero_grad()
    out[0, pred_idx].backward()

    # Compute heat map
    grads   = gradients[0].squeeze()           # (C, H, W)
    acts    = activations[0].squeeze()          # (C, H, W)
    weights = grads.mean(dim=(1, 2))            # (C,)
    cam     = (weights[:, None, None] * acts).sum(0)
    cam     = F.relu(cam)
    cam     = cam.numpy()
    cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_img = np.uint8(255 * cam)
    cam_pil = Image.fromarray(cam_img).resize(
        (224, 224), Image.BILINEAR
    )
    cam_arr = np.array(cam_pil)

    # Overlay
    orig    = np.array(img_pil.resize((224, 224)))
    heat    = mpl_cm.jet(cam_arr / 255.0)[:, :, :3]
    overlay = np.clip(0.5 * orig / 255.0 + 0.5 * heat, 0, 1)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(cam_arr, cmap='jet')
    axes[1].set_title('Grad-CAM Heat Map')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title(
        f'Overlay\nPredicted: {pred_cls}\nConfidence: {prob*100:.1f}%'
    )
    axes[2].axis('off')

    plt.suptitle(f'Grad-CAM — {pred_cls}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return pred_cls, prob

# ── Run on one sample per species ────────────────────────────────────
IMAGE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data', 'insect_images'
)

print("\nGenerating Grad-CAM for each species:")
for species in class_names:
    folder = os.path.join(IMAGE_DIR, species)
    if not os.path.exists(folder):
        print(f"  {species}: folder not found — skipping")
        continue

    imgs = [f for f in os.listdir(folder)
            if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if not imgs:
        print(f"  {species}: no images found — skipping")
        continue

    sample    = os.path.join(folder, imgs[0])
    save_path = f'outputs/gradcam/{species}_gradcam.png'

    pred, conf = run_gradcam(sample, save_path)
    print(f"  {species}: Predicted={pred}  "
          f"Confidence={conf*100:.1f}%  → {save_path}")

print("\n  Heat maps saved in: outputs/gradcam/")
print("\n✓ DONE — gradcam.py complete")
print("  Next: python fusion_engine.py")
