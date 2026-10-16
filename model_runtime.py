from __future__ import annotations

import io
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn


IMAGE_SIZE = 128
MODEL_VERSION = 2
ModelMode = Literal["manual", "full_image", "classification"]

MODEL_FILES: dict[ModelMode, str] = {
    "manual": "segmentation_classifier.pt",
    "full_image": "full_image_segmentation.pt",
    "classification": "image_classifier.pt",
}


class ImageClassifier(nn.Module):
    def __init__(self, class_count: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, class_count),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.features(image)


class SegmentationClassifier(nn.Module):
    def __init__(self, class_count: int) -> None:
        super().__init__()
        self.encoder1 = self._block(3, 16)
        self.encoder2 = self._block(16, 32)
        self.encoder3 = self._block(32, 64)
        self.bottleneck = self._block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = self._block(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = self._block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.decoder1 = self._block(32, 16)
        self.mask_head = nn.Conv2d(16, 1, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, class_count),
        )

    @staticmethod
    def _block(input_channels: int, output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature1 = self.encoder1(image)
        feature2 = self.encoder2(self.pool(feature1))
        feature3 = self.encoder3(self.pool(feature2))
        bottleneck = self.bottleneck(self.pool(feature3))
        decoded3 = self.decoder3(torch.cat((self.up3(bottleneck), feature3), dim=1))
        decoded2 = self.decoder2(torch.cat((self.up2(decoded3), feature2), dim=1))
        decoded1 = self.decoder1(torch.cat((self.up1(decoded2), feature1), dim=1))
        return self.classifier(bottleneck), self.mask_head(decoded1)


def load_model(
    model_dir: Path, mode: ModelMode
) -> tuple[ImageClassifier | SegmentationClassifier, list[str], int]:
    model_path = model_dir / MODEL_FILES[mode]
    if not model_path.is_file():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {model_path}. "
            "GitHub 저장소의 outputs 폴더에 모델 파일을 포함했는지 확인하세요."
        )

    expected_type = "classification" if mode == "classification" else "segmentation"
    model, classes, image_size, model_type = load_model_from_bytes(
        model_path.read_bytes(), model_path.name
    )
    if model_type != expected_type:
        raise ValueError(f"선택한 모델 유형과 파일 내용이 일치하지 않습니다: {model_path.name}")
    return model, classes, image_size


def load_model_from_bytes(
    checkpoint_data: bytes, model_name: str
) -> tuple[ImageClassifier | SegmentationClassifier, list[str], int, str]:
    try:
        checkpoint = torch.load(
            io.BytesIO(checkpoint_data), map_location="cpu", weights_only=True
        )
    except (pickle.UnpicklingError, EOFError, RuntimeError) as error:
        raise ValueError(f"모델 파일을 안전하게 읽을 수 없습니다: {model_name}") from error

    if not isinstance(checkpoint, dict) or checkpoint.get("version") != MODEL_VERSION:
        raise ValueError(f"지원하지 않거나 올바르지 않은 모델 파일입니다: {model_name}")

    model_type = checkpoint.get("model_type", "segmentation")
    if model_type not in ("classification", "segmentation"):
        raise ValueError(f"모델 유형을 확인할 수 없습니다: {model_name}")

    classes = checkpoint.get("classes")
    image_size = checkpoint.get("image_size")
    state_dict = checkpoint.get("state_dict")
    if (
        not isinstance(classes, list)
        or len(classes) < 2
        or not all(isinstance(name, str) and name for name in classes)
        or len(set(classes)) != len(classes)
        or not isinstance(image_size, int)
        or isinstance(image_size, bool)
        or image_size < 8
        or image_size > 512
        or image_size % 8 != 0
        or not isinstance(state_dict, dict)
    ):
        raise ValueError(f"모델 파일의 클래스/가중치 정보가 올바르지 않습니다: {model_name}")

    model: ImageClassifier | SegmentationClassifier
    if model_type == "classification":
        model = ImageClassifier(len(classes))
    else:
        model = SegmentationClassifier(len(classes))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        raise ValueError(f"모델 가중치 구조가 지원되지 않습니다: {model_name}") from error
    model.eval()
    return model, classes, image_size, model_type


def predict_image(
    source: Image.Image,
    model: ImageClassifier | SegmentationClassifier,
    classes: list[str],
    image_size: int,
) -> tuple[str, float, np.ndarray | None]:
    image = ImageOps.exif_transpose(source).convert("RGB")
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy()).unsqueeze(0)

    with torch.inference_mode():
        if isinstance(model, ImageClassifier):
            class_logits = model(tensor)
            mask: np.ndarray | None = None
        else:
            class_logits, mask_logits = model(tensor)
            mask = (
                torch.sigmoid(mask_logits)[0, 0] >= 0.5
            ).cpu().numpy()

        probabilities = class_logits.softmax(dim=1)[0]
        predicted_index = int(probabilities.argmax().item())
        confidence = float(probabilities[predicted_index].item())

    if mask is not None:
        mask_image = Image.fromarray(np.uint8(mask) * 255).resize(
            image.size, Image.Resampling.NEAREST
        )
        mask = np.asarray(mask_image) > 0
    return classes[predicted_index], confidence, mask


def open_uploaded_image(data: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as source:
            return ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError("지원하지 않거나 손상된 이미지 파일입니다.") from error


def render_mask_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    original = np.asarray(image.convert("RGB"), dtype=np.float32)
    red = np.zeros_like(original)
    red[..., 0] = 255
    overlay = original.copy()
    overlay[mask] = overlay[mask] * 0.45 + red[mask] * 0.55
    return Image.fromarray(np.uint8(overlay))
