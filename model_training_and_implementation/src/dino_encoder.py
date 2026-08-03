import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DINOv2ImageEncoder:
    """
    Python 3.9-compatible frozen DINOv2-S image encoder.

    Returns a 384-dimensional CLS-token embedding using the same
    predict(image)["embedding"] interface as the ResNet encoder.
    """

    def __init__(
        self,
        pretrained: bool = True,
        device: str = "cuda",
        l2_normalize: bool = True,
    ):
        if not pretrained:
            raise ValueError(
                "DINOv2ImageEncoder currently supports pretrained=True only."
            )

        self.device = torch.device(device)
        self.l2_normalize = bool(l2_normalize)
        self.model_id = "facebook/dinov2-small"

        self.processor = AutoImageProcessor.from_pretrained(
            self.model_id
        )

        self.model = AutoModel.from_pretrained(
            self.model_id,
            use_safetensors=True,
        ).to(self.device)

        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> dict:
        if image is None:
            raise ValueError("The input image cannot be None.")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Expected a BGR image with shape (height, width, 3), "
                f"but received {image.shape}."
            )

        # Images read by OpenCV are BGR. The Hugging Face processor expects RGB.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        inputs = self.processor(
            images=pil_image,
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"].to(self.device)

        outputs = self.model(pixel_values=pixel_values)

        # Token zero is the global CLS token.
        # DINOv2-S produces a 384-dimensional representation.
        embedding = outputs.last_hidden_state[:, 0, :]

        if self.l2_normalize:
            embedding = F.normalize(
                embedding,
                p=2,
                dim=1,
            )

        embedding = (
            embedding.squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return {
            "embedding": embedding,
        }