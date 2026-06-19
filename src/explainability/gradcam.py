"""
Grad-CAM — Explainable AI
=========================
মুখের কোন অংশ দেখে model decision নিল, সেটা heatmap দিয়ে দেখায়।
এটা তোমার project এর "explainability" USP — defence এ স্যাররা এটা পছন্দ করে।

এটা Phase 6 এর কাজ। Face CNN train হওয়ার পর এটা চালাবে।
"""

import torch
import torch.nn.functional as F


class GradCAM:
    """
    একটা CNN এর শেষ conv layer এর উপর Grad-CAM heatmap বানায়।
    """

    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx=None):
        # input_tensor: (1, 1, H, W)
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # weights = global-average-pool of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from src.models.face_cnn import FaceCNN

    model = FaceCNN(num_classes=7)
    # শেষ conv layer = features[8] (Block 3 এর Conv2d)
    target = model.features[8]
    cam_gen = GradCAM(model, target)
    dummy = torch.randn(1, 1, 48, 48, requires_grad=True)
    cam, idx = cam_gen.generate(dummy)
    print(f"Grad-CAM heatmap shape: {cam.shape}, predicted class: {idx}")
