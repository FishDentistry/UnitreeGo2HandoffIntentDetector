import inspect

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class DINOObjectDetector:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        confidence: float = 0.25,
    ):
        self.model_id = model_id
        self.confidence = confidence
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(model_id)
            .to(self.device)
        )

    def _get_target_size(self, img):
        if isinstance(img, Image.Image):
            return img.size[::-1]

        if hasattr(img, "shape") and len(img.shape) >= 2:
            return (int(img.shape[0]), int(img.shape[1]))

        raise TypeError(
            f"Unsupported image type for target size: {type(img)}"
        )

    def normalize_dino_labels(self, labels):
        if (
            isinstance(labels, list)
            and labels
            and isinstance(labels[0], list)
        ):
            labels = labels[0]

        if isinstance(labels, list):
            labels = ". ".join(
                str(x).strip().rstrip(".")
                for x in labels
                if str(x).strip()
            )

        if isinstance(labels, str):
            labels = labels.strip()

            if not labels:
                raise ValueError("No DINO labels provided.")

            labels = labels.rstrip(".").strip() + "."
            return labels

        raise TypeError(
            f"Unsupported labels format: {type(labels)}"
        )

    def _post_process_grounded_object_detection(
        self,
        outputs,
        input_ids,
        target_sizes,
    ):
        post_process_fn = (
            self.processor.post_process_grounded_object_detection
        )

        parameters = inspect.signature(
            post_process_fn
        ).parameters

        kwargs = {
            "text_threshold": 0.25,
            "target_sizes": target_sizes,
        }

        if "threshold" in parameters:
            kwargs["threshold"] = self.confidence
        elif "box_threshold" in parameters:
            kwargs["box_threshold"] = self.confidence
        else:
            raise RuntimeError(
                "Unsupported Transformers Grounding DINO API: "
                "post_process_grounded_object_detection() has neither "
                "'threshold' nor 'box_threshold'."
            )

        return post_process_fn(
            outputs,
            input_ids,
            **kwargs,
        )

    def predict(self, img, class_names):
        class_names = self.normalize_dino_labels(class_names)

        inputs = self.processor(
            images=img,
            text=class_names,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self._post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            target_sizes=[self._get_target_size(img)],
        )

        if isinstance(results, list):
            if not results:
                return []

            results = results[0]

        detections = []

        labels = results.get(
            "text_labels",
            results["labels"],
        )

        for box, score, label in zip(
            results["boxes"],
            results["scores"],
            labels,
        ):
            label = str(label).strip()

            if not label:
                continue

            detections.append(
                {
                    "label": label,
                    "score": float(score.item()),
                    "box_xyxy": [
                        float(v)
                        for v in box.tolist()
                    ],
                }
            )

        return detections