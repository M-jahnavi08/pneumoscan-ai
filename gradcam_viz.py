# gradcam_viz.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.efficientnet_b3(weights=None)
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_features, 2)
)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)
        cam     = cam.squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        probs   = output.softmax(dim=1)[0].detach().cpu().numpy()
        return cam, class_idx, probs


def explain_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    raw_img  = Image.open(img_path).convert("RGB")
    input_t  = transform(raw_img).unsqueeze(0).to(device)
    input_t.requires_grad_()

    target_layer = model.features[-1]
    grad_cam     = GradCAM(model, target_layer)
    cam, pred_idx, probs = grad_cam.generate(input_t)

    img_np  = np.array(raw_img.resize((224, 224)))
    heatmap = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Prediction: {CLASS_NAMES[pred_idx]}  "
        f"(Normal: {probs[0]:.1%}, Pneumonia: {probs[1]:.1%})",
        fontsize=14, fontweight='bold',
        color='green' if pred_idx == 0 else 'red'
    )

    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title("Original X-Ray")
    axes[0].axis('off')

    axes[1].imshow(heatmap[:, :, ::-1])
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')

    axes[2].imshow(overlay[:, :, ::-1])
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("gradcam_result.png", dpi=150)
    plt.show()
    print(f"\n✅ Prediction: {CLASS_NAMES[pred_idx]} ({probs[pred_idx]:.1%} confidence)")


# ── Run it ──────────────────────────────────────────────────────────────────
# Step 1: Open this folder in File Explorer:
#         C:\Users\JAHNAVI SATYA\chest-xray-ai\chest_xray\test\PNEUMONIA
# Step 2: Copy any filename you see (e.g. person1_virus_6.jpeg)
# Step 3: Paste it below replacing the filename after PNEUMONIA/

explain_image("chest_xray/train/PNEUMONIA/person1_bacteria_1.jpeg")