import cv2
import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision import models
from torchvision.models import ResNet18_Weights


class ResNet18ImageEncoder:
    """
    Small wrapper for extracting fixed ResNet-18 image embeddings.

    Expected input:
        image: OpenCV BGR image from cv2.imread(...)

    Returns:
        {
            "embedding": np.ndarray shape (512,), dtype float32,
            "embedding_dim": 512,
        }
    """

    def __init__(
        self,
        pretrained: bool = True,
        device: str = "auto",
        l2_normalize: bool = False,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            self.preprocess = weights.transforms()
        else:
            weights = None
            # Still use ImageNet-style preprocessing shape.
            self.preprocess = ResNet18_Weights.DEFAULT.transforms()

        self.model = models.resnet18(weights=weights)

        # Replace final classifier with identity so forward() returns 512-D features.
        self.model.fc = nn.Identity()

        self.model = self.model.to(self.device)
        self.model.eval()

        self.l2_normalize = bool(l2_normalize)


    @torch.inference_mode()
    def predict(self, image_bgr):
        if image_bgr is None:
            raise ValueError("image_bgr is None")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        x = self.preprocess(image_pil)
        x = x.unsqueeze(0).to(self.device)

        embedding = self.model(x)
        embedding = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)

        if self.l2_normalize:
            denom = float(np.linalg.norm(embedding)) + 1e-8
            embedding = embedding / denom

        return {
            "embedding": embedding,
            "embedding_dim": int(embedding.shape[0]),
        }