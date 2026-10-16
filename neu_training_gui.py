from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator

import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from model_runtime import ImageClassifier, SegmentationClassifier


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
ANNOTATION_DIR = OUTPUT_DIR / "image_masks"
BATCH_ANNOTATION_DIR = OUTPUT_DIR / "image_masks_full"
MODEL_PATH = OUTPUT_DIR / "segmentation_classifier.pt"
CLASSIFICATION_MODEL_PATH = OUTPUT_DIR / "image_classifier.pt"
BATCH_SEGMENTATION_MODEL_PATH = OUTPUT_DIR / "full_image_segmentation.pt"
CLASSIFICATION_LABELS_PATH = OUTPUT_DIR / "image_labels.csv"
RESULTS_PATH = OUTPUT_DIR / "test_predictions.csv"
PREDICTED_MASK_DIR = OUTPUT_DIR / "predicted_masks"
PREDICTED_FULL_MASK_DIR = OUTPUT_DIR / "predicted_masks_full"
DEFAULT_TRAINING_FOLDER = ROOT / "data" / "neu" / "train"
DEFAULT_TEST_FOLDER = ROOT / "data" / "neu" / "test" / "new"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
IMAGE_SIZE = 128
MODEL_VERSION = 2
torch.set_num_threads(min(4, torch.get_num_threads()))


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def resolve_folder(folder: str, label: str) -> Path:
    folder = folder.strip()
    if not folder:
        raise ValueError(f"{label} 폴더 경로를 입력하세요.")
    path = Path(folder).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"{label} 폴더를 찾을 수 없습니다: {path}")
    return path


def image_files(folder: Path, recursive: bool = False) -> list[Path]:
    candidates = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        (
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: str(path).casefold(),
    )


def discover_training_data(folder: str) -> tuple[Path, list[str], dict[str, int]]:
    root = resolve_folder(folder, "학습 데이터")
    class_folders = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    if len(class_folders) < 2:
        raise ValueError(
            "학습 폴더 바로 아래에 클래스별 하위 폴더가 2개 이상 있어야 합니다. "
            "예: train\\scratch\\image1.jpg, train\\dent\\image2.jpg"
        )
    counts = {path.name: len(image_files(path, recursive=True)) for path in class_folders}
    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        raise ValueError(f"이미지가 없는 클래스 폴더: {', '.join(empty)}")
    return root, [path.name for path in class_folders], counts


def mask_path_for(
    image_path: Path,
    training_root: Path,
    mask_root: Path = ANNOTATION_DIR,
) -> Path:
    try:
        relative = image_path.resolve().relative_to(training_root.resolve())
    except ValueError as error:
        raise ValueError(f"학습 이미지 폴더 바깥의 경로입니다: {image_path}") from error
    mask_path = (mask_root / relative).with_suffix(".mask.png").resolve()
    if not mask_path.is_relative_to(mask_root.resolve()):
        raise ValueError(f"마스크 경로가 저장 폴더 밖을 가리킵니다: {mask_path}")
    return mask_path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_signatures(
    root: Path,
    paths: list[str],
    mask_root: Path = ANNOTATION_DIR,
) -> dict[str, str | None]:
    signatures: dict[str, str | None] = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        relative = path.relative_to(root.resolve()).as_posix()
        mask_path = mask_path_for(path, root, mask_root)
        signatures[relative] = (
            f"{file_sha256(path)}:{file_sha256(mask_path)}"
            if mask_path.is_file()
            else None
        )
    return signatures


def checkpoint_training_signatures() -> tuple[str | None, dict[str, str]]:
    if not MODEL_PATH.is_file():
        return None, {}
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    training_root = checkpoint.get("training_root")
    signatures = checkpoint.get("training_signatures", {})
    if not isinstance(training_root, str) or not isinstance(signatures, dict):
        return None, {}
    return training_root, signatures


def label_table(root: Path, paths: list[str]) -> pd.DataFrame:
    saved_root, saved_signatures = checkpoint_training_signatures()
    current_signatures = training_signatures(root, paths)
    rows = []
    for raw_path in paths:
        path = Path(raw_path)
        relative = path.relative_to(root).as_posix()
        signature = current_signatures[relative]
        trained = (
            signature is not None
            and saved_root == str(root.resolve())
            and saved_signatures.get(relative) == signature
        )
        rows.append(
            {
                "클래스 폴더": path.relative_to(root).parts[0],
                "이미지": path.name,
                "라벨 상태": "완료" if signature is not None else "미완료",
                "학습 상태": "완료" if trained else "미학습/수정됨",
            }
        )
    return pd.DataFrame(rows)


def label_folder_status(table: pd.DataFrame) -> str:
    total = len(table)
    labeled = int((table["라벨 상태"] == "완료").sum())
    trained = int((table["학습 상태"] == "완료").sum())
    counts = table.groupby("클래스 폴더").agg(
        이미지=("이미지", "size"),
        라벨완료=("라벨 상태", lambda values: int((values == "완료").sum())),
    )
    per_class = ", ".join(
        f"{name}: {row['라벨완료']}/{row['이미지']}"
        for name, row in counts.iterrows()
    )
    return (
        f"이미지 **{total}장** · 라벨링 **{labeled}/{total}장** · "
        f"현재 저장 모델에 반영 **{trained}/{total}장**  \n"
        f"클래스별 라벨링 완료: {per_class}"
    )


def scan_label_folder(folder: str) -> tuple[Any, ...]:
    root, classes, counts = discover_training_data(folder)
    paths = [
        str(image_path)
        for class_name in classes
        for image_path in image_files(root / class_name, recursive=True)
    ]
    choices = [str(root / class_name) for class_name in classes]
    table = label_table(root, paths)
    status = label_folder_status(table)
    class_counts = ", ".join(f"{name}: {count}장" for name, count in counts.items())
    return (
        f"{status}  \n클래스 수: **{len(classes)}종** · {class_counts}  \n"
        "이미지 목록을 불러왔습니다. 표에서 사진을 클릭하면 라벨링 화면에서 열립니다.",
        paths,
        gr.update(choices=choices, value=choices[0]),
        gr.update(choices=paths, value=None),
        table,
    )


def update_label_images(class_folder: str, all_paths: list[str]) -> Any:
    prefix = str(Path(class_folder).resolve())
    choices = [
        path
        for path in all_paths
        if Path(path).resolve().parent == Path(prefix)
        or Path(prefix) in Path(path).resolve().parents
    ]
    return gr.update(choices=choices, value=None)


def select_label_table_row(
    event: gr.SelectData, all_paths: list[str], training_folder: str
) -> tuple[Any, Any, str]:
    if not all_paths:
        raise ValueError("먼저 학습 폴더의 이미지 목록을 불러오세요.")
    selected = event.index[0] if isinstance(event.index, (tuple, list)) else event.index
    if not isinstance(selected, int) or not 0 <= selected < len(all_paths):
        raise ValueError(f"선택한 이미지 행을 찾을 수 없습니다: {event.index}")
    image_path = all_paths[selected]
    editor, status = load_image_for_labeling(image_path, training_folder)
    return gr.update(value=image_path), editor, status


def load_image_for_labeling(image_path: str, training_folder: str) -> tuple[Any, ...]:
    root = resolve_folder(training_folder, "학습 데이터")
    path = Path(image_path).resolve()
    mask_path = mask_path_for(path, root)
    try:
        background = load_rgb_image(path)
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"이미지를 열 수 없습니다: {path}") from error

    layer = Image.new("RGBA", background.size, (0, 0, 0, 0))
    if mask_path.is_file():
        with Image.open(mask_path) as saved:
            mask = saved.convert("L").resize(background.size, Image.Resampling.NEAREST)
        alpha = np.asarray(mask, dtype=np.uint8)
        overlay = np.zeros((background.height, background.width, 4), dtype=np.uint8)
        overlay[..., 0] = 255
        overlay[..., 1] = 25
        overlay[..., 2] = 25
        overlay[..., 3] = np.where(alpha > 0, 150, 0).astype(np.uint8)
        layer = Image.fromarray(overlay, mode="RGBA")
        coverage = float(np.count_nonzero(alpha)) / alpha.size
        status = (
            f"저장된 마스크를 불러왔습니다. 빨간색 영역이 기존 라벨입니다. "
            f"표시 면적 {coverage:.1%}. 브러시로 추가하거나 지우개로 수정한 뒤 저장하세요."
        )
    else:
        status = (
            "새 이미지입니다. 브러시로 결함 부분을 칠하세요. 정상 이미지라면 "
            "아무것도 칠하지 않고 빈 마스크를 저장할 수 있습니다."
        )
    editor_value = {
        "background": background,
        "layers": [layer],
        "composite": background,
    }
    return editor_value, status


def extract_binary_mask(editor_value: Any, image_size: tuple[int, int]) -> np.ndarray:
    if not isinstance(editor_value, dict):
        raise ValueError("편집된 브러시 마스크가 없습니다. 먼저 이미지를 불러오세요.")

    layers = editor_value.get("layers") or []
    if not layers:
        raise ValueError("브러시 레이어를 찾을 수 없습니다.")
    width, height = image_size
    combined = np.zeros((height, width), dtype=bool)
    for layer in layers:
        if layer is None:
            continue
        if isinstance(layer, Image.Image):
            layer_image = layer.convert("RGBA")
        elif isinstance(layer, (str, Path)):
            with Image.open(layer) as source:
                layer_image = source.convert("RGBA")
        else:
            array = np.asarray(layer)
            if array.ndim == 2:
                alpha = array
            elif array.ndim == 3 and array.shape[2] == 4:
                alpha = array[..., 3]
            elif array.ndim == 3 and array.shape[2] == 3:
                alpha = np.max(array, axis=2)
            else:
                raise ValueError(f"알 수 없는 브러시 레이어 모양입니다: {array.shape}")
            layer_mask = Image.fromarray(np.uint8(alpha > 0) * 255).resize(
                (width, height), Image.Resampling.NEAREST
            )
            combined |= np.asarray(layer_mask) > 0
            continue

        if layer_image.size != (width, height):
            layer_image = layer_image.resize((width, height), Image.Resampling.NEAREST)
        combined |= np.asarray(layer_image.getchannel("A")) > 0
    return np.uint8(combined) * 255


def save_image_annotation(
    image_path: str, training_folder: str, editor_value: Any
) -> tuple[str, Any, str, pd.DataFrame]:
    root = resolve_folder(training_folder, "학습 데이터")
    path = Path(image_path).resolve()
    mask_path = mask_path_for(path, root)
    try:
        image = load_rgb_image(path)
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"이미지를 열 수 없습니다: {path}") from error

    mask = extract_binary_mask(editor_value, image.size)
    if not np.any(mask):
        class_name = path.relative_to(root).parts[0]
        if class_name.casefold() not in {"good", "normal", "ok", "background", "normal_image"}:
            raise ValueError(
                "저장할 결함 영역이 비어 있습니다. 결함 이미지라면 브러시로 표시하고, "
                "정상 이미지라면 클래스 폴더 이름을 good 또는 normal로 사용하세요."
            )

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=mask_path.parent, suffix=".mask.png", delete=False
        ) as temporary_file:
            temporary_path = temporary_file.name
        Image.fromarray(mask, mode="L").save(temporary_path, format="PNG")
        os.replace(temporary_path, mask_path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)

    class_name = path.relative_to(root).parts[0]
    coverage = float(np.count_nonzero(mask)) / mask.size
    saved_layer = np.zeros((*mask.shape, 4), dtype=np.uint8)
    saved_layer[..., 0] = 255
    saved_layer[..., 1] = 25
    saved_layer[..., 2] = 25
    saved_layer[..., 3] = np.where(mask > 0, 150, 0).astype(np.uint8)
    editor_result = {
        "background": image,
        "layers": [Image.fromarray(saved_layer, mode="RGBA")],
        "composite": image,
    }
    _, classes, _ = discover_training_data(str(root))
    all_paths = [
        str(image)
        for name in classes
        for image in image_files(root / name, recursive=True)
    ]
    table = label_table(root, all_paths)
    status = label_folder_status(table)
    return (
        f"마스크 저장 완료: `{display_path(mask_path)}` · 결함 표시 면적 {coverage:.1%}",
        editor_result,
        status,
        table,
    )


def generate_batch_labels(training_folder: str) -> str:
    root, classes, counts = discover_training_data(training_folder)
    rows: list[dict[str, str]] = []
    total = sum(counts.values())
    for class_name in classes:
        for image_path in image_files(root / class_name, recursive=True):
            try:
                image = load_rgb_image(image_path)
            except (OSError, UnidentifiedImageError) as error:
                raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}") from error

            mask_path = mask_path_for(image_path, root, BATCH_ANNOTATION_DIR)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=mask_path.parent, suffix=".mask.png", delete=False
                ) as temporary_file:
                    temporary_path = temporary_file.name
                Image.new("L", image.size, color=255).save(
                    temporary_path, format="PNG"
                )
                os.replace(temporary_path, mask_path)
            finally:
                if temporary_path and os.path.exists(temporary_path):
                    os.unlink(temporary_path)

            rows.append(
                {
                    "이미지": image_path.relative_to(root).as_posix(),
                    "분류 라벨": class_name,
                    "결함 마스크": str(mask_path),
                }
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_csv: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=OUTPUT_DIR, suffix=".csv", delete=False
        ) as temporary_file:
            temporary_csv = temporary_file.name
        pd.DataFrame(rows).to_csv(
            temporary_csv, index=False, encoding="utf-8-sig"
        )
        os.replace(temporary_csv, CLASSIFICATION_LABELS_PATH)
    finally:
        if temporary_csv and os.path.exists(temporary_csv):
            os.unlink(temporary_csv)

    count_summary = ", ".join(f"{name}: {count}장" for name, count in counts.items())
    return (
        f"일괄 라벨 생성 완료: **{total}장** · 클래스 {len(classes)}종  \n"
        f"전체 이미지 결함 마스크: `{display_path(BATCH_ANNOTATION_DIR)}`  \n"
        f"클래스 분류 라벨 목록: `{display_path(CLASSIFICATION_LABELS_PATH)}`  \n"
        f"클래스별 이미지: {count_summary}. 수동 마스크 폴더와 별도로 저장했습니다."
    )


class SegmentationDataset(Dataset[tuple[torch.Tensor, int, torch.Tensor]]):
    def __init__(
        self,
        root: Path,
        classes: list[str],
        mask_root: Path = ANNOTATION_DIR,
    ) -> None:
        self.root = root
        self.classes = classes
        self.mask_root = mask_root
        self.samples = [
            (path, class_index, mask_path_for(path, root, mask_root))
            for class_index, class_name in enumerate(classes)
            for path in image_files(root / class_name, recursive=True)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        path, class_index, mask_path = self.samples[index]
        try:
            image = load_rgb_image(path).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
            )
            with Image.open(mask_path) as source:
                mask = source.convert("L").resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST
                )
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(f"학습 이미지/마스크를 읽을 수 없습니다: {path}") from error

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1).copy())
        mask_tensor = torch.from_numpy(mask_array[None, ...].copy())
        return image_tensor, class_index, mask_tensor


class ImageClassificationDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, root: Path, classes: list[str]) -> None:
        self.samples = [
            (path, class_index)
            for class_index, class_name in enumerate(classes)
            for path in image_files(root / class_name, recursive=True)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, class_index = self.samples[index]
        try:
            image = load_rgb_image(path).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
            )
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(f"학습 이미지를 읽을 수 없습니다: {path}") from error
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1).copy())
        return image_tensor, class_index


def split_samples(
    dataset: Dataset[Any], class_count: int
) -> tuple[list[int], list[int]]:
    train_indices: list[int] = []
    validation_indices: list[int] = []
    grouped: list[list[int]] = [[] for _ in range(class_count)]
    for index, sample in enumerate(dataset.samples):
        label = sample[1]
        grouped[label].append(index)

    generator = torch.Generator().manual_seed(42)
    for indices in grouped:
        if len(indices) < 2:
            raise ValueError("학습/검증 분할을 위해 클래스마다 이미지가 최소 2장 필요합니다.")
        order = torch.randperm(len(indices), generator=generator).tolist()
        validation_count = min(len(indices) - 1, max(1, round(len(indices) * 0.2)))
        validation_indices.extend(indices[position] for position in order[:validation_count])
        train_indices.extend(indices[position] for position in order[validation_count:])
    return train_indices, validation_indices


def save_checkpoint(
    model: nn.Module,
    classes: list[str],
    training_root: Path,
    signatures: dict[str, str | None],
    model_path: Path = MODEL_PATH,
    model_type: str = "segmentation",
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=model_path.parent, suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_path = temporary_file.name
        torch.save(
            {
                "version": MODEL_VERSION,
                "model_type": model_type,
                "image_size": IMAGE_SIZE,
                "classes": classes,
                "state_dict": model.state_dict(),
                "training_root": str(training_root.resolve()),
                "training_signatures": signatures,
            },
            temporary_path,
        )
        os.replace(temporary_path, model_path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def train_model(
    training_folder: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    mask_root: Path = ANNOTATION_DIR,
    model_path: Path = MODEL_PATH,
    training_name: str = "수동 영역",
) -> Iterator[tuple[str, pd.DataFrame]]:
    if not 1 <= int(epochs) <= 100:
        raise ValueError("Epoch는 1~100 사이여야 합니다.")
    if int(batch_size) not in (1, 2, 4, 8, 16):
        raise ValueError("Batch size는 1, 2, 4, 8, 16 중 하나여야 합니다.")
    if not np.isfinite(float(learning_rate)) or not 0 < float(learning_rate) <= 0.1:
        raise ValueError("Learning rate는 0보다 크고 0.1 이하여야 합니다.")

    root, classes, counts = discover_training_data(training_folder)
    dataset = SegmentationDataset(root, classes, mask_root)
    missing = [
        path
        for path, _, mask_path in dataset.samples
        if not mask_path.is_file()
    ]
    if missing:
        preview = "\n".join(f"- {path}" for path in missing[:10])
        remainder = (
            f"\n외 {len(missing) - 10}장" if len(missing) > 10 else ""
        )
        raise ValueError(
            f"{training_name} 학습 전에 모든 이미지의 마스크를 생성해야 합니다. "
            f"미라벨 이미지 {len(missing)}장:\n{preview}{remainder}"
        )

    train_indices, validation_indices = split_samples(dataset, len(classes))
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
    )

    torch.manual_seed(42)
    model = SegmentationClassifier(len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    classification_loss = nn.CrossEntropyLoss()
    segmentation_loss = nn.BCEWithLogitsLoss()
    history: list[dict[str, Any]] = []
    yield (
        f"데이터 로드: `{root}` · 클래스 {len(classes)}종 · 전체 {len(dataset)}장. "
        f"{training_name} 마스크와 클래스 분류를 학습합니다.",
        pd.DataFrame(history),
    )

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        for images, labels, masks in train_loader:
            optimizer.zero_grad(set_to_none=True)
            class_logits, mask_logits = model(images)
            class_loss = classification_loss(class_logits, labels)
            mask_loss = segmentation_loss(mask_logits, masks)
            loss = class_loss + mask_loss
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(labels)
            train_correct += int((class_logits.argmax(dim=1) == labels).sum().item())

        model.eval()
        validation_loss_sum = 0.0
        validation_correct = 0
        intersection = 0
        union = 0
        with torch.no_grad():
            for images, labels, masks in validation_loader:
                class_logits, mask_logits = model(images)
                class_loss = classification_loss(class_logits, labels)
                mask_loss = segmentation_loss(mask_logits, masks)
                validation_loss_sum += (class_loss + mask_loss).item() * len(labels)
                validation_correct += int((class_logits.argmax(dim=1) == labels).sum().item())
                predictions = torch.sigmoid(mask_logits) >= 0.5
                targets = masks >= 0.5
                intersection += int((predictions & targets).sum().item())
                union += int((predictions | targets).sum().item())

        train_count = len(train_indices)
        validation_count = len(validation_indices)
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_count,
            "train_accuracy": train_correct / train_count,
            "val_loss": validation_loss_sum / validation_count,
            "val_accuracy": validation_correct / validation_count,
            "val_mask_iou": intersection / union if union else 1.0,
        }
        history.append(metrics)
        yield (
            f"학습 중: **{epoch}/{int(epochs)} epoch** · "
            f"Validation 분류 정확도 {metrics['val_accuracy']:.1%} · "
            f"마스크 IoU {metrics['val_mask_iou']:.3f}",
            pd.DataFrame(history),
        )

    signatures = training_signatures(
        root,
        [str(path) for path, _, _ in dataset.samples],
        mask_root,
    )
    save_checkpoint(
        model,
        classes,
        root,
        signatures,
        model_path=model_path,
        model_type="segmentation",
    )
    count_summary = ", ".join(f"{name}: {count}장" for name, count in counts.items())
    yield (
        f"학습 완료. 모델 저장: `{display_path(model_path)}`  \n"
        f"폴더별 이미지: {count_summary}  \n"
        f"마지막 Validation 분류 정확도: **{history[-1]['val_accuracy']:.1%}** · "
        f"마스크 IoU: **{history[-1]['val_mask_iou']:.3f}**. "
        "두 지표는 내부 검증 결과이며 새 이미지 성능을 보장하지 않습니다.",
        pd.DataFrame(history),
    )


def train_classification_model(
    training_folder: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    model_path: Path = CLASSIFICATION_MODEL_PATH,
) -> Iterator[tuple[str, pd.DataFrame]]:
    if not 1 <= int(epochs) <= 100:
        raise ValueError("Epoch는 1~100 사이여야 합니다.")
    if int(batch_size) not in (1, 2, 4, 8, 16):
        raise ValueError("Batch size는 1, 2, 4, 8, 16 중 하나여야 합니다.")
    if not np.isfinite(float(learning_rate)) or not 0 < float(learning_rate) <= 0.1:
        raise ValueError("Learning rate는 0보다 크고 0.1 이하여야 합니다.")

    root, classes, counts = discover_training_data(training_folder)
    dataset = ImageClassificationDataset(root, classes)
    train_indices, validation_indices = split_samples(dataset, len(classes))
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
    )

    torch.manual_seed(42)
    model = ImageClassifier(len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    loss_function = nn.CrossEntropyLoss()
    history: list[dict[str, Any]] = []
    yield (
        f"분류 전용 데이터 로드: `{root}` · 클래스 {len(classes)}종 · "
        f"전체 {len(dataset)}장. 마스크 없이 폴더명을 정답 라벨로 학습합니다.",
        pd.DataFrame(history),
    )

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(labels)
            train_correct += int((logits.argmax(dim=1) == labels).sum().item())

        model.eval()
        validation_loss_sum = 0.0
        validation_correct = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                logits = model(images)
                validation_loss_sum += loss_function(logits, labels).item() * len(labels)
                validation_correct += int((logits.argmax(dim=1) == labels).sum().item())

        train_count = len(train_indices)
        validation_count = len(validation_indices)
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_count,
            "train_accuracy": train_correct / train_count,
            "val_loss": validation_loss_sum / validation_count,
            "val_accuracy": validation_correct / validation_count,
        }
        history.append(metrics)
        yield (
            f"분류 학습 중: **{epoch}/{int(epochs)} epoch** · "
            f"검증 정확도 {metrics['val_accuracy']:.1%}",
            pd.DataFrame(history),
        )

    signatures = {
        Path(path).relative_to(root).as_posix(): file_sha256(Path(path))
        for path, _ in dataset.samples
    }
    save_checkpoint(
        model,
        classes,
        root,
        signatures,
        model_path=model_path,
        model_type="classification",
    )
    count_summary = ", ".join(f"{name}: {count}장" for name, count in counts.items())
    yield (
        f"분류 전용 학습 완료. 모델 저장: `{display_path(model_path)}`  \n"
        f"폴더별 이미지: {count_summary} · 검증 정확도 "
        f"**{history[-1]['val_accuracy']:.1%}**. 새 이미지 성능을 보장하지 않습니다.",
        pd.DataFrame(history),
    )


def load_model(
    model_path: Path = MODEL_PATH,
) -> tuple[SegmentationClassifier, list[str], int]:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"학습 모델이 없습니다: {model_path}\n"
            "마스크 라벨링을 마친 뒤 모델 학습을 실행하세요."
        )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != MODEL_VERSION:
        raise ValueError("저장된 모델 버전이 현재 앱과 호환되지 않습니다. 다시 학습하세요.")
    classes = checkpoint.get("classes")
    image_size = checkpoint.get("image_size")
    if (
        not isinstance(classes, list)
        or len(classes) < 2
        or not all(isinstance(name, str) for name in classes)
        or not isinstance(image_size, int)
    ):
        raise ValueError("저장된 모델의 클래스 정보가 올바르지 않습니다.")
    model = SegmentationClassifier(len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, classes, image_size


def load_classification_model() -> tuple[ImageClassifier, list[str], int]:
    if not CLASSIFICATION_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"분류 전용 모델이 없습니다: {CLASSIFICATION_MODEL_PATH}\n"
            "먼저 분류 전용 학습을 실행하세요."
        )
    checkpoint = torch.load(
        CLASSIFICATION_MODEL_PATH, map_location="cpu", weights_only=True
    )
    if (
        checkpoint.get("version") != MODEL_VERSION
        or checkpoint.get("model_type") != "classification"
    ):
        raise ValueError("저장된 분류 모델이 현재 앱과 호환되지 않습니다. 다시 학습하세요.")
    classes = checkpoint.get("classes")
    image_size = checkpoint.get("image_size")
    if (
        not isinstance(classes, list)
        or len(classes) < 2
        or not all(isinstance(name, str) for name in classes)
        or not isinstance(image_size, int)
    ):
        raise ValueError("저장된 분류 모델의 클래스 정보가 올바르지 않습니다.")
    model = ImageClassifier(len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, classes, image_size


def train_batch_segmentation_model(
    training_folder: str, epochs: int, batch_size: int, learning_rate: float
) -> Iterator[tuple[str, pd.DataFrame]]:
    yield from train_model(
        training_folder,
        epochs,
        batch_size,
        learning_rate,
        mask_root=BATCH_ANNOTATION_DIR,
        model_path=BATCH_SEGMENTATION_MODEL_PATH,
        training_name="전체 이미지",
    )


def load_selected_model(
    model_mode: str,
) -> tuple[SegmentationClassifier | ImageClassifier, list[str], int]:
    if model_mode == "manual":
        return load_model(MODEL_PATH)
    if model_mode == "full_image":
        return load_model(BATCH_SEGMENTATION_MODEL_PATH)
    if model_mode == "classification":
        return load_classification_model()
    raise ValueError(f"알 수 없는 모델 유형입니다: {model_mode}")


def overlay_predicted_mask(
    image: Image.Image,
    mask: np.ndarray,
    prediction_label: str = "AI DEFECT PREDICTION",
) -> Image.Image:
    original = image.convert("RGB")
    binary_mask = Image.fromarray(np.uint8(mask) * 255).resize(
        original.size, Image.Resampling.NEAREST
    )
    alpha = np.asarray(binary_mask) > 0
    overlay = np.asarray(original).copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    overlay[alpha] = (
        overlay[alpha].astype(np.float32) * 0.45
        + red[alpha].astype(np.float32) * 0.55
    ).astype(np.uint8)

    panel_width, panel_height, gap = 420, 360, 16
    header_height = 48
    comparison = Image.new(
        "RGB",
        (panel_width * 2 + gap, panel_height),
        "#07111f",
    )
    draw = ImageDraw.Draw(comparison)
    font = ImageFont.load_default(size=16)
    labels = ("ORIGINAL IMAGE", prediction_label)
    panel_positions = (0, panel_width + gap)
    image_area = (panel_width - 32, panel_height - header_height - 24)
    display_size = ImageOps.contain(
        original, image_area, method=Image.Resampling.LANCZOS
    ).size
    masked = Image.fromarray(overlay).resize(
        display_size, Image.Resampling.LANCZOS
    )
    displayed_original = original.resize(display_size, Image.Resampling.LANCZOS)

    for index, panel_x in enumerate(panel_positions):
        draw.rounded_rectangle(
            (panel_x, 0, panel_x + panel_width - 1, panel_height - 1),
            radius=12,
            fill="#0e1d2e",
            outline="#29435d",
            width=1,
        )
        draw.text(
            (panel_x + 16, 16),
            labels[index],
            fill="#dcecff",
            font=font,
        )
        draw.line(
            (panel_x + 16, header_height - 1, panel_x + panel_width - 16, header_height - 1),
            fill="#29435d",
            width=1,
        )

    image_y = header_height + (panel_height - header_height - display_size[1]) // 2
    image_x_offset = (panel_width - display_size[0]) // 2
    comparison.paste(
        displayed_original,
        (image_x_offset, image_y),
    )
    comparison.paste(
        masked,
        (panel_width + gap + image_x_offset, image_y),
    )
    return comparison


def predict_one(
    image_path: Path,
    model: SegmentationClassifier | ImageClassifier,
    classes: list[str],
    image_size: int,
    test_root: Path,
    model_mode: str = "manual",
) -> tuple[dict[str, Any], Image.Image]:
    try:
        original = load_rgb_image(image_path)
        resized = original.resize((image_size, image_size), Image.Resampling.BILINEAR)
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}") from error

    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy()).unsqueeze(0)
    with torch.no_grad():
        if isinstance(model, ImageClassifier):
            class_logits = model(tensor)
            mask = np.zeros((image_size, image_size), dtype=bool)
        else:
            class_logits, mask_logits = model(tensor)
            mask = (torch.sigmoid(mask_logits)[0, 0] >= 0.5).cpu().numpy()
        probabilities = class_logits.softmax(dim=1)[0]
        predicted_index = int(probabilities.argmax().item())

    relative_path = image_path.relative_to(test_root)
    output_mask: Path | None = None
    if isinstance(model, SegmentationClassifier):
        output_root = (
            PREDICTED_FULL_MASK_DIR
            if model_mode == "full_image"
            else PREDICTED_MASK_DIR
        )
        output_mask = (output_root / relative_path).with_suffix(".mask.png")
        output_mask.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.uint8(mask) * 255).resize(
            original.size, Image.Resampling.NEAREST
        ).save(output_mask, format="PNG")
    row = {
        "이미지": str(relative_path),
        "분류 결과": classes[predicted_index],
        "분류 점수(보정되지 않음)": f"{float(probabilities[predicted_index].item()):.1%}",
        "결함 표시 면적": f"{float(mask.mean()):.1%}" if output_mask else "미실행",
        "예측 마스크": str(output_mask) if output_mask else "분류 전용",
    }
    return row, overlay_predicted_mask(
        original,
        mask,
        "MASK NOT RUN" if isinstance(model, ImageClassifier) else "AI DEFECT PREDICTION",
    )


def classify_test_folder(
    test_folder: str, model_mode: str = "manual"
) -> tuple[str, pd.DataFrame, list[str], Any, Any]:
    folder = resolve_folder(test_folder, "분석할 이미지")
    paths = image_files(folder, recursive=True)
    if not paths:
        raise ValueError(
            f"테스트 폴더에서 이미지 파일을 찾을 수 없습니다: {folder}\n"
            f"지원 확장자: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )
    model, classes, image_size = load_selected_model(model_mode)

    rows: list[dict[str, Any]] = []
    for path in paths:
        row, _ = predict_one(path, model, classes, image_size, folder, model_mode)
        rows.append(row)

    result = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    mask_summary = (
        f"예측 마스크 저장: `{display_path(PREDICTED_MASK_DIR if model_mode == 'manual' else PREDICTED_FULL_MASK_DIR)}`  \n"
        if model_mode != "classification"
        else "분류 전용 모델을 사용했습니다. 마스크 예측은 실행하지 않았습니다.  \n"
    )
    result_description = (
        "분류 결과를 표에 표시합니다. 분류 전용 모델은 결함 마스크를 계산하지 않습니다."
        if model_mode == "classification"
        else "결과 목록에서 행을 선택하면 원본 이미지와 결함 예측 영역이 아래에 표시됩니다. "
        "빨간색은 모델이 결함으로 예측한 픽셀입니다."
    )
    result_caution = (
        " 분류 점수는 보정된 확률이 아닙니다."
        if model_mode == "classification"
        else " 분류 점수는 보정된 확률이 아니며 예측 마스크도 정답 보장이 아닙니다."
    )
    summary = (
        f"분류 완료: **{len(result)}장** · "
        f"{mask_summary}"
        f"{result_description}{result_caution}"
    )
    choices = result["이미지"].tolist()
    first_row = rows[0]
    first_image_path = folder / first_row["이미지"]
    _, comparison = predict_one(
        first_image_path, model, classes, image_size, folder, model_mode
    )
    detail = (
        f"**{first_row['이미지']}** · 분류: **{first_row['분류 결과']}** · "
        f"표시 면적: {first_row['결함 표시 면적']}"
    )
    return summary, result, [str(path) for path in paths], comparison, detail


def show_selected_test_image(
    event: gr.SelectData,
    image_paths: list[str],
    test_folder: str,
    model_mode: str = "manual",
) -> tuple[Any, str]:
    if not image_paths:
        raise ValueError("먼저 테스트 이미지 분류를 실행하세요.")
    selected = event.index[0] if isinstance(event.index, (tuple, list)) else event.index
    if not isinstance(selected, int) or not 0 <= selected < len(image_paths):
        raise ValueError(f"선택한 이미지 행을 찾을 수 없습니다: {event.index}")

    root = resolve_folder(test_folder, "테스트")
    model, classes, image_size = load_selected_model(model_mode)
    path = Path(image_paths[selected]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("선택한 이미지가 테스트 폴더 밖에 있습니다.")
    row, comparison = predict_one(path, model, classes, image_size, root, model_mode)
    detail = (
        f"**{row['이미지']}** · 분류: **{row['분류 결과']}** · "
        f"점수: {row['분류 점수(보정되지 않음)']} · "
        f"결함 표시 면적: {row['결함 표시 면적']}"
    )
    return comparison, detail


APP_CSS = """
body, .gradio-container {
    background: #07111f !important;
    color: #e6f1ff;
}

.gradio-container {
    max-width: 1440px !important;
    margin: 0 auto !important;
    padding: 28px clamp(16px, 4vw, 52px) 48px !important;
    background:
        radial-gradient(ellipse at 8% 0%, rgba(0, 217, 255, 0.09), transparent 32%),
        radial-gradient(ellipse at 95% 18%, rgba(107, 91, 255, 0.08), transparent 28%),
        #07111f !important;
}

#app-hero {
    position: relative;
    overflow: hidden;
    margin: 0 0 24px;
    padding: clamp(24px, 4vw, 44px);
    border: 1px solid rgba(86, 160, 198, 0.24);
    border-radius: 24px;
    background: linear-gradient(120deg, rgba(13, 30, 49, 0.97), rgba(10, 20, 39, 0.94));
    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.25);
}

#app-hero::after {
    position: absolute;
    top: -145px;
    right: -70px;
    width: 360px;
    height: 360px;
    border: 1px solid rgba(0, 217, 255, 0.14);
    border-radius: 50%;
    box-shadow: 0 0 0 34px rgba(0, 217, 255, 0.025), 0 0 0 72px rgba(0, 217, 255, 0.02);
    content: "";
    pointer-events: none;
}

.hero-top {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
}

.hero-kicker, .hero-step small {
    color: #7fa5c3;
    font-family: ui-monospace, Consolas, monospace;
    font-size: 11px;
    letter-spacing: 0.16em;
}

.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    color: #62e5ff;
}

.hero-kicker::before {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #39e4c5;
    box-shadow: 0 0 14px rgba(57, 228, 197, 0.8);
    content: "";
}

.hero-title {
    margin: 18px 0 10px;
    color: #f2f8ff;
    font-size: clamp(30px, 4vw, 48px);
    font-weight: 750;
    letter-spacing: -0.045em;
    line-height: 1.18;
}

.hero-title span {
    color: #53dafa;
}

.hero-description {
    max-width: 720px;
    margin: 0;
    color: #a5b9d0;
    font-size: 15px;
    line-height: 1.8;
}

.hero-orbit {
    display: grid;
    flex: 0 0 136px;
    width: 136px;
    height: 136px;
    place-items: center;
    border: 1px solid rgba(73, 206, 246, 0.3);
    border-radius: 50%;
    background: radial-gradient(circle, rgba(17, 93, 124, 0.26), rgba(8, 22, 39, 0.16) 66%);
    box-shadow: inset 0 0 34px rgba(0, 217, 255, 0.08), 0 0 42px rgba(0, 217, 255, 0.08);
}

.hero-orbit span {
    color: #8cecff;
    font-family: ui-monospace, Consolas, monospace;
    font-size: 13px;
    letter-spacing: 0.08em;
    text-align: center;
}

.hero-flow {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 30px;
}

.hero-step {
    display: grid;
    grid-template-columns: auto 1fr;
    align-items: center;
    column-gap: 12px;
    padding: 13px 15px;
    border: 1px solid rgba(91, 137, 169, 0.2);
    border-radius: 14px;
    background: rgba(5, 16, 30, 0.46);
}

.hero-step b {
    color: #dcecff;
    font-size: 14px;
    font-weight: 650;
}

.hero-step small {
    grid-column: 2;
    margin-top: 3px;
    font-size: 9px;
    letter-spacing: 0.12em;
}

.hero-step-number {
    display: grid;
    grid-row: span 2;
    width: 34px;
    height: 34px;
    place-items: center;
    border: 1px solid rgba(65, 212, 244, 0.3);
    border-radius: 10px;
    background: rgba(42, 186, 221, 0.1);
    color: #69e2fb;
    font-family: ui-monospace, Consolas, monospace;
    font-size: 12px;
}

.tabs {
    gap: 0 !important;
}

.tab-nav {
    gap: 8px !important;
    border-bottom: 1px solid rgba(92, 137, 168, 0.2) !important;
}

.tab-nav button {
    border: 1px solid transparent !important;
    border-radius: 12px 12px 0 0 !important;
    color: #91a9c2 !important;
    transition: color 160ms ease, background 160ms ease, border-color 160ms ease;
}

.tab-nav button.selected {
    border-color: rgba(69, 204, 238, 0.24) !important;
    background: rgba(33, 154, 187, 0.12) !important;
    color: #6ee6ff !important;
}

.tabitem {
    padding: clamp(16px, 3vw, 28px) !important;
    border: 1px solid rgba(92, 137, 168, 0.2) !important;
    border-top: 0 !important;
    border-radius: 0 0 20px 20px !important;
    background: rgba(11, 24, 41, 0.82) !important;
}

.tabitem p {
    color: #d2e1ef !important;
    font-size: 15px;
    line-height: 1.75;
}

.tabitem code {
    padding: 2px 6px;
    border: 1px solid #29435d;
    border-radius: 5px;
    background: #162a40 !important;
    color: #a5efff !important;
    font-size: 0.92em;
}

.tabitem label.float {
    border: 1px solid rgba(83, 218, 250, 0.3) !important;
    border-radius: 8px !important;
    background: #173149 !important;
    color: #b4effa !important;
}

.tabitem button:not(.primary) {
    border: 1px solid #2a425d !important;
    border-radius: 11px !important;
    background: #12243a !important;
    color: #dcecff !important;
}

.tabitem button:not(.primary):hover {
    border-color: rgba(83, 218, 250, 0.55) !important;
    background: #17314b !important;
}

.table-wrap .header-cell {
    border-color: #2a4058 !important;
    background: #14263a !important;
    color: #c8d9eb !important;
}

.table-wrap td {
    border-color: rgba(69, 99, 128, 0.36) !important;
    background: #0d1b2c !important;
    color: #dcecff !important;
}

.table-wrap tbody tr:nth-child(even) td {
    background: #122238 !important;
}

.table-wrap [role="gridcell"] {
    background: #0d1b2c !important;
    color: #dcecff !important;
}

.table-wrap [role="row"]:nth-child(even) [role="gridcell"] {
    background: #122238 !important;
}

.table-wrap tbody td .cell-wrap,
.table-wrap [role="gridcell"] .cell-wrap {
    color: #dcecff !important;
    font-size: 14px;
}

button.primary {
    box-shadow: 0 7px 22px rgba(0, 199, 232, 0.16);
    font-weight: 700 !important;
}

input, textarea, select {
    color-scheme: dark;
}

@media (max-width: 700px) {
    .hero-orbit {
        display: none;
    }

    .hero-flow {
        grid-template-columns: 1fr;
        gap: 8px;
        margin-top: 22px;
    }

    .tab-nav {
        overflow-x: auto;
    }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        scroll-behavior: auto !important;
        transition-duration: 0.01ms !important;
    }
}
"""

APP_THEME = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size="lg",
).set(
    body_background_fill="#07111f",
    body_background_fill_dark="#07111f",
    body_text_color="#e6f1ff",
    body_text_color_dark="#e6f1ff",
    body_text_size="16px",
    body_text_color_subdued="#93a9c4",
    body_text_color_subdued_dark="#93a9c4",
    background_fill_primary="#0d192b",
    background_fill_primary_dark="#0d192b",
    background_fill_secondary="#111f33",
    background_fill_secondary_dark="#111f33",
    block_background_fill="#0c192b",
    block_background_fill_dark="#0c192b",
    block_border_color="#233752",
    block_border_color_dark="#233752",
    input_background_fill="#091626",
    input_background_fill_dark="#091626",
    input_border_color="#2a3d57",
    input_border_color_dark="#2a3d57",
    button_primary_background_fill="#13c7e8",
    button_primary_background_fill_dark="#13c7e8",
    button_primary_background_fill_hover="#38dafa",
    button_primary_background_fill_hover_dark="#38dafa",
    button_primary_text_color="#04121d",
    button_primary_text_color_dark="#04121d",
)


def make_app() -> gr.Blocks:
    with gr.Blocks(
        title="NEU VISION | 결함 분석 스튜디오",
        theme=APP_THEME,
        css=APP_CSS,
        fill_width=True,
    ) as app:
        gr.HTML(
            """
            <header id="app-hero">
                <div class="hero-top">
                    <div>
                        <div class="hero-kicker">NEU / VISION SYSTEMS&nbsp;&nbsp;·&nbsp;&nbsp;AI INSPECTION STUDIO</div>
                        <h1 class="hero-title">결함을 발견하고,<br><span>정밀하게 학습합니다.</span></h1>
                        <p class="hero-description">
                            이미지 라벨링부터 모델 학습, 예측 검증까지.
                            하나의 워크플로에서 제조 품질 검사를 경험하세요.
                        </p>
                    </div>
                    <div class="hero-orbit" aria-hidden="true"><span>VISION<br>AI / 01</span></div>
                </div>
                <div class="hero-flow" aria-label="작업 순서">
                    <div class="hero-step"><span class="hero-step-number">01</span><b>결함 영역 라벨링</b><small>PIXEL ANNOTATION</small></div>
                    <div class="hero-step"><span class="hero-step-number">02</span><b>모델 학습</b><small>MODEL TRAINING</small></div>
                    <div class="hero-step"><span class="hero-step-number">03</span><b>검사 결과 확인</b><small>INFERENCE &amp; REVIEW</small></div>
                </div>
            </header>
            """
        )
        with gr.Tab("01 · 결함 라벨링", elem_classes=["workflow-tab"]):
            training_folder = gr.Textbox(
                label="학습 데이터 폴더 경로",
                value=str(DEFAULT_TRAINING_FOLDER),
                placeholder=r"예: D:\datasets\train",
                info="바로 아래 클래스 폴더를 둡니다. 예: train\\scratch\\01.jpg, train\\dent\\01.jpg",
            )
            scan_button = gr.Button("학습 폴더의 이미지 목록 불러오기")
            gr.Markdown(
                "수동 마스크는 아래 편집기에서 이미지별로 그립니다. 전체 이미지를 같은 라벨로 처리할 경우 "
                "일괄 생성 기능을 사용하세요. 일괄 생성은 클래스 폴더명을 분류 라벨로 기록하고, "
                "전체 픽셀 결함 마스크를 수동 마스크와 다른 폴더에 저장합니다."
            )
            batch_label_button = gr.Button(
                "전체 이미지 분류 라벨 + 결함 마스크 일괄 생성"
            )
            batch_label_status = gr.Markdown()
            label_folder_state = gr.State([])
            label_status = gr.Markdown()
            label_table_view = gr.Dataframe(
                value=pd.DataFrame(
                    columns=["클래스 폴더", "이미지", "라벨 상태", "학습 상태"]
                ),
                label="클래스별 이미지 라벨링/학습 상태 — 행을 클릭하면 이미지 열기",
                interactive=False,
            )
            with gr.Row():
                label_class = gr.Dropdown(
                    choices=[], label="클래스 폴더", interactive=True
                )
                label_image = gr.Dropdown(
                    choices=[], label="라벨링할 이미지", interactive=True, filterable=True
                )
            load_label_button = gr.Button("선택 이미지 불러오기")
            gr.Markdown(
                "이미지가 표시되면 브러시로 결함 픽셀을 칠하세요. 저장된 빨간 영역은 기존 마스크입니다. "
                "정상 이미지는 폴더 이름을 `good` 또는 `normal`로 만들고 빈 마스크를 저장하세요. "
                "라벨은 `outputs/image_masks`에 저장되며 원본 이미지는 변경하지 않습니다."
            )
            annotation_editor = gr.ImageEditor(
                value=None,
                type="pil",
                label="이미지 위에서 결함 영역 칠하기",
                brush=gr.Brush(
                    default_size=18,
                    colors=["#ff1919"],
                    default_color="#ff1919",
                    color_mode="fixed",
                ),
                eraser=gr.Eraser(default_size=24),
                layers=True,
                transforms=[],
                image_mode="RGBA",
                height=600,
            )
            save_annotation_button = gr.Button("현재 이미지 라벨 저장", variant="primary")
            annotation_status = gr.Markdown()

            scan_button.click(
                fn=scan_label_folder,
                inputs=[training_folder],
                outputs=[
                    label_status,
                    label_folder_state,
                    label_class,
                    label_image,
                    label_table_view,
                ],
            )
            label_table_view.select(
                fn=select_label_table_row,
                inputs=[label_folder_state, training_folder],
                outputs=[label_image, annotation_editor, annotation_status],
            )
            label_class.change(
                fn=update_label_images,
                inputs=[label_class, label_folder_state],
                outputs=[label_image],
            )
            load_label_button.click(
                fn=load_image_for_labeling,
                inputs=[label_image, training_folder],
                outputs=[annotation_editor, annotation_status],
            )
            save_annotation_button.click(
                fn=save_image_annotation,
                inputs=[label_image, training_folder, annotation_editor],
                outputs=[
                    annotation_status,
                    annotation_editor,
                    label_status,
                    label_table_view,
                ],
            )
            batch_label_button.click(
                fn=generate_batch_labels,
                inputs=[training_folder],
                outputs=[batch_label_status],
            )

        with gr.Tab("02 · 모델 학습", elem_classes=["workflow-tab"]):
            gr.Markdown(
                "각 학습 버튼은 별도의 데이터·목적·모델 파일을 사용합니다. "
                "수동 마스크 학습은 모든 학습 이미지에 브러시 마스크가 필요합니다."
            )
            train_folder = gr.Textbox(
                label="라벨링을 완료한 학습 데이터 폴더",
                value=str(DEFAULT_TRAINING_FOLDER),
                placeholder=r"예: D:\datasets\train",
            )
            gr.Markdown(
                "**Epoch**는 학습 이미지 전체를 한 번씩 학습하는 횟수입니다. "
                "Batch size는 한 번에 처리할 이미지 수입니다.  \n"
                "모델은 이미지 분류와 픽셀 단위 결함 마스크를 함께 학습합니다. "
                "마스크 IoU는 예측 영역과 직접 그린 정답 영역의 겹침 정도입니다."
            )
            with gr.Row():
                epochs = gr.Slider(1, 100, value=15, step=1, label="Epoch")
                batch_size = gr.Dropdown(
                    choices=[1, 2, 4, 8, 16], value=4, label="Batch size"
                )
                learning_rate = gr.Number(
                    value=0.001, label="Learning rate", minimum=0.00001, maximum=0.1
                )
            train_button = gr.Button("수동 마스크 기반 분할 모델 학습", variant="primary")
            training_status = gr.Markdown()
            training_history = gr.Dataframe(label="Epoch별 검증 결과", interactive=False)
            train_event = train_button.click(
                fn=train_model,
                inputs=[train_folder, epochs, batch_size, learning_rate],
                outputs=[training_status, training_history],
            )
            train_event.then(
                fn=scan_label_folder,
                inputs=[train_folder],
                outputs=[
                    label_status,
                    label_folder_state,
                    label_class,
                    label_image,
                    label_table_view,
                ],
            )
            gr.Markdown("### 분류 전용 모델")
            gr.Markdown(
                "클래스 폴더명을 정답 라벨로 사용해 이미지 분류만 학습합니다. "
                "마스크를 읽거나 생성하지 않으며 분할 모델과 별도 파일로 저장합니다."
            )
            classification_train_button = gr.Button(
                "마스크 없이 이미지 분류 모델 학습", variant="primary"
            )
            classification_training_status = gr.Markdown()
            classification_training_history = gr.Dataframe(
                label="분류 전용 Epoch별 검증 결과", interactive=False
            )
            classification_train_button.click(
                fn=train_classification_model,
                inputs=[train_folder, epochs, batch_size, learning_rate],
                outputs=[
                    classification_training_status,
                    classification_training_history,
                ],
            )

            gr.Markdown("### 전체 이미지 마스크 기반 분할 모델")
            gr.Markdown(
                f"일괄 생성한 전체 이미지 마스크 `{display_path(BATCH_ANNOTATION_DIR)}`만 사용합니다. "
                "수동 브러시 마스크는 사용하지 않으며, 분류 전용 모델 및 수동 분할 모델과 따로 저장합니다."
            )
            batch_segmentation_train_button = gr.Button(
                "전체 이미지 마스크로 별도 분할 모델 학습", variant="primary"
            )
            batch_segmentation_training_status = gr.Markdown()
            batch_segmentation_training_history = gr.Dataframe(
                label="전체 이미지 마스크 Epoch별 검증 결과", interactive=False
            )
            batch_segmentation_train_button.click(
                fn=train_batch_segmentation_model,
                inputs=[train_folder, epochs, batch_size, learning_rate],
                outputs=[
                    batch_segmentation_training_status,
                    batch_segmentation_training_history,
                ],
            )

        with gr.Tab("03 · 테스트", elem_classes=["workflow-tab"]):
            gr.Markdown(
                "테스트 폴더 안의 이미지를 실행 시 분류합니다. 상단 결과 목록의 행을 클릭하면 "
                "해당 행에 맞는 원본 이미지와 예측 결함 영역이 아래에 표시됩니다."
            )
            test_folder = gr.Textbox(
                label="테스트 이미지 폴더 경로",
                value=str(DEFAULT_TEST_FOLDER),
                placeholder=r"예: D:\datasets\test",
            )
            inference_model = gr.Dropdown(
                choices=[
                    ("수동 마스크 기반 모델", "manual"),
                    ("전체 이미지 마스크 모델", "full_image"),
                    ("분류 전용 모델 (마스크 예측 없음)", "classification"),
                ],
                value="manual",
                label="검사에 사용할 모델",
            )
            classify_button = gr.Button(
                "테스트 이미지 분류 및 결함 영역 표시", variant="primary"
            )
            classification_status = gr.Markdown()
            classification_results = gr.Dataframe(
                label="결과 이미지 목록 — 이미지 행을 클릭하세요",
                interactive=False,
            )
            test_image_state = gr.State([])
            selected_image_detail = gr.Markdown()
            selected_image_preview = gr.Image(
                label="선택한 이미지 · 원본 / 결함 예측 영역",
                type="pil",
                interactive=False,
            )
            classify_button.click(
                fn=classify_test_folder,
                inputs=[test_folder, inference_model],
                outputs=[
                    classification_status,
                    classification_results,
                    test_image_state,
                    selected_image_preview,
                    selected_image_detail,
                ],
            )
            classification_results.select(
                fn=show_selected_test_image,
                inputs=[test_image_state, test_folder, inference_model],
                outputs=[selected_image_preview, selected_image_detail],
            )
            gr.Markdown(
                f"수동 마스크 모델: `{display_path(MODEL_PATH)}` · "
                f"수동 마스크: `{display_path(ANNOTATION_DIR)}`  \n"
                f"전체 마스크 모델: `{display_path(BATCH_SEGMENTATION_MODEL_PATH)}` · "
                f"전체 마스크: `{display_path(BATCH_ANNOTATION_DIR)}`  \n"
                f"분류 전용 모델: `{display_path(CLASSIFICATION_MODEL_PATH)}` · "
                f"분류 라벨 목록: `{display_path(CLASSIFICATION_LABELS_PATH)}`  \n"
                f"예측 마스크: `{display_path(PREDICTED_MASK_DIR)}` / "
                f"`{display_path(PREDICTED_FULL_MASK_DIR)}`"
            )
    return app


if __name__ == "__main__":
    make_app().launch(server_name="127.0.0.1", share=False, inbrowser=True)
