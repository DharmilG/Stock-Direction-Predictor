"""GradCAM on an in-memory image.

Heavy dependencies are imported lazily so importing the API works even before
the CV dependency overlay is installed or a checkpoint exists.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def get_gradcam(image: Image.Image, model=None):
    """Return a 224x224x3 heatmap overlay for the supplied PIL image."""
    import numpy as np
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from torchvision import transforms

    if model is None:
        from ..api.inference import load_cv_model

        model = load_cv_model()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    rgb_image = image.convert("RGB")
    input_tensor = transform(rgb_image).unsqueeze(0)

    class PatternOnlyModel(torch.nn.Module):
        def __init__(self, full_model):
            super().__init__()
            self.full_model = full_model

        def forward(self, x):
            return self.full_model.pattern_head(self.full_model.backbone(x))

    pattern_model = PatternOnlyModel(model)
    pattern_model.eval()
    target_layer = model.backbone.layer4[-1]

    original_requires_grad = [parameter.requires_grad for parameter in model.backbone.parameters()]
    try:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = True
        cam = GradCAM(model=pattern_model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    finally:
        for parameter, original in zip(model.backbone.parameters(), original_requires_grad):
            parameter.requires_grad = original

    rgb_array = np.array(rgb_image.resize((224, 224))) / 255.0
    return show_cam_on_image(rgb_array.astype(np.float32), grayscale_cam, use_rgb=True)


def get_gradcam_for_path(image_path: Path | str, model=None):
    with Image.open(image_path) as image:
        return get_gradcam(image.convert("RGB"), model=model)


__all__ = ["get_gradcam", "get_gradcam_for_path"]
