from __future__ import annotations

import base64
import hashlib
import io
import os
from pathlib import Path
from typing import Literal, Union

from PIL import Image, ImageOps


Provider = Literal["openai", "gemini"]
ImageInput = Union[str, Path, bytes, bytearray, memoryview]

HANDOFF_PROMPT = """
You are a binary visual classifier running on a robot's front-facing camera.

Determine whether the image depicts a person who is CURRENTLY TRYING TO
INITIATE A HANDOFF OF AN OBJECT TO THE CAMERA/ROBOT.

Return 1 only when:
- A person is holding an identifiable object, and
- The person is intentionally presenting, offering, or extending that object
  toward the camera/robot as though expecting the robot to take it.

Merely holding, carrying, using, examining, or displaying an object is not
necessarily a handoff. A hand extended without an object is not an object
handoff. Passing an object to another visible person is not a handoff to the
camera/robot. Use arm posture, object position, body orientation, and direction
of presentation as evidence. If the situation is unclear, return 0.

Output exactly one character and no explanation:
1 = handoff initiation
0 = not a handoff initiation
""".strip()

DEFAULT_MODELS = {
    "openai": "gpt-5.4-mini",
    "gemini": "gemini-2.5-flash",
}


class CloudHandoffClassifier:
    """Binary handoff classifier backed by OpenAI or Gemini."""

    def __init__(
        self,
        provider: Provider,
        model: str | None = None,
        api_key: str | None = None,
        max_image_dimension: int = 1280,
        jpeg_quality: int = 90,
    ) -> None:
        provider = provider.lower()
        if provider not in DEFAULT_MODELS:
            raise ValueError("provider must be 'openai' or 'gemini'")
        if max_image_dimension <= 0:
            raise ValueError("max_image_dimension must be positive")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")

        self.provider: Provider = provider  # type: ignore[assignment]
        self.model = model or DEFAULT_MODELS[provider]
        self.max_image_dimension = max_image_dimension
        self.jpeg_quality = jpeg_quality
        self.prompt_sha256 = hashlib.sha256(
            HANDOFF_PROMPT.encode("utf-8")
        ).hexdigest()

        if provider == "openai":
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("Install OpenAI support with: pip install openai") from exc

            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("Set OPENAI_API_KEY or pass api_key")
            self.client = OpenAI(api_key=key)
        else:
            try:
                from google import genai
            except ImportError as exc:
                raise ImportError("Install Gemini support with: pip install google-genai") from exc

            key = api_key or os.getenv("GEMINI_API_KEY")
            if not key:
                raise ValueError("Set GEMINI_API_KEY or pass api_key")
            self.client = genai.Client(api_key=key)

    def predict(self, image: ImageInput) -> int:
        prediction, _ = self.predict_with_response(image)
        return prediction

    def predict_with_response(self, image: ImageInput) -> tuple[int, str]:
        image_bytes, mime_type = self._prepare_image(image)

        if self.provider == "openai":
            raw_response = self._predict_openai(image_bytes, mime_type)
        else:
            raw_response = self._predict_gemini(image_bytes, mime_type)

        return self._parse_binary_answer(raw_response), raw_response

    def _predict_openai(self, image_bytes: bytes, mime_type: str) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": HANDOFF_PROMPT,
                        },
                        {
                            "type": "input_image",
                            "image_url": (
                                f"data:{mime_type};base64,{encoded}"
                            ),
                            "detail": "high",
                        },
                    ],
                }
            ],

            # This task does not need extended hidden reasoning.
            reasoning={"effort": "none"},

            # Leave enough room for the visible response.
            max_output_tokens=128,
        )

        if not response.output_text:
            incomplete_reason = None

            if response.incomplete_details is not None:
                incomplete_reason = response.incomplete_details.reason

            raise RuntimeError(
                "OpenAI returned no visible text. "
                f"status={response.status!r}, "
                f"incomplete_reason={incomplete_reason!r}, "
                f"usage={response.usage!r}, "
                f"output={response.output!r}"
            )

        return response.output_text

    def _predict_gemini(self, image_bytes: bytes, mime_type: str) -> str:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                HANDOFF_PROMPT,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=32,
                response_mime_type="text/plain",
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini returned no text")
        return response.text

    def _prepare_image(self, image: ImageInput) -> tuple[bytes, str]:
        try:
            if isinstance(image, (str, Path)):
                with Image.open(image) as loaded:
                    prepared = loaded.copy()
            elif isinstance(image, (bytes, bytearray, memoryview)):
                with Image.open(io.BytesIO(bytes(image))) as loaded:
                    prepared = loaded.copy()
            else:
                raise TypeError("image must be a path or encoded image bytes")
        except (OSError, ValueError) as exc:
            raise ValueError("The supplied image could not be decoded") from exc

        prepared = ImageOps.exif_transpose(prepared)
        prepared.thumbnail(
            (self.max_image_dimension, self.max_image_dimension),
            Image.Resampling.LANCZOS,
        )
        if prepared.mode != "RGB":
            prepared = prepared.convert("RGB")

        output = io.BytesIO()
        prepared.save(
            output,
            format="JPEG",
            quality=self.jpeg_quality,
            optimize=True,
        )
        return output.getvalue(), "image/jpeg"

    @staticmethod
    def _parse_binary_answer(answer: str) -> int:
        normalized = answer.strip().lower()
        normalized = normalized.removeprefix("```text").removeprefix("```")
        normalized = normalized.removesuffix("```").strip()
        normalized = normalized.strip("\"'").rstrip(".!,:;").strip()

        if normalized in {"1", "yes", "true"}:
            return 1
        if normalized in {"0", "no", "false"}:
            return 0
        raise RuntimeError(f"Invalid binary response: {answer!r}")