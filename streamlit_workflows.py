from __future__ import annotations

import hashlib
import io
import os
import shutil
import stat
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

import neu_training_gui as training
from model_runtime import (
    MODEL_FILES,
    ModelMode,
    load_model_from_bytes,
    load_model,
    predict_image,
    render_mask_overlay,
)


MAX_ARCHIVE_BYTES = 500 * 1024 * 1024
MAX_EXTRACTED_BYTES = 1024 * 1024 * 1024
MAX_ARCHIVE_FILES = 20_000
MAX_TEST_IMAGES = 20_000
MASK_CLASS_NAMES = {"good", "normal", "ok", "background", "normal_image"}


def workspace_dir() -> Path:
    workspace_id = st.session_state.setdefault("workspace_id", uuid.uuid4().hex)
    root = training.OUTPUT_DIR / "streamlit_sessions" / workspace_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def _extract_dataset_archive(data: bytes, name: str, key: str) -> Path:
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError("ZIP 파일은 500MB 이하만 업로드할 수 있습니다.")
    digest = hashlib.sha256(data).hexdigest()
    workspace = workspace_dir()
    destination = workspace / f"{key}-{digest[:16]}"
    marker = workspace / f".{key}-{digest[:16]}.complete"
    if marker.is_file() and destination.is_dir():
        return _normalize_dataset_root(destination)

    stage = workspace / f".extract-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as error:
            raise ValueError(f"{name}: 정상적인 ZIP 파일이 아닙니다.") from error
        with archive:
            entries = []
            for entry in archive.infolist():
                member = PurePosixPath(entry.filename)
                if (
                    entry.is_dir()
                    or "__MACOSX" in member.parts
                    or member.name == ".DS_Store"
                ):
                    continue
                entries.append(entry)
            if len(entries) > MAX_ARCHIVE_FILES:
                raise ValueError(
                    f"ZIP에 파일이 너무 많습니다 ({len(entries):,}개). "
                    f"최대 {MAX_ARCHIVE_FILES:,}개까지 지원합니다."
                )
            total_bytes = sum(entry.file_size for entry in entries)
            if total_bytes > MAX_EXTRACTED_BYTES:
                raise ValueError("압축 해제 후 데이터는 1GB 이하만 지원합니다.")

            root = stage.resolve()
            for entry in entries:
                member = PurePosixPath(entry.filename)
                if (
                    member.is_absolute()
                    or not member.parts
                    or any(part in ("", ".", "..") for part in member.parts)
                    or ":" in member.parts[0]
                ):
                    raise ValueError(f"ZIP에 안전하지 않은 파일 경로가 있습니다: {entry.filename}")
                mode = entry.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError(f"ZIP 심볼릭 링크는 허용되지 않습니다: {entry.filename}")
                if entry.flag_bits & 0x1:
                    raise ValueError(f"암호화된 ZIP 항목은 지원하지 않습니다: {entry.filename}")
                target = (stage / Path(*member.parts)).resolve()
                if not target.is_relative_to(root):
                    raise ValueError(f"ZIP 경로가 대상 폴더 밖을 가리킵니다: {entry.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(entry) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)

        if destination.exists():
            raise ValueError(
                "이 세션에 이전에 추출이 완료되지 않은 같은 데이터가 있습니다. "
                "ZIP을 다시 올려 새 세션에서 시도하세요."
            )
        os.replace(stage, destination)
        marker.write_text("complete", encoding="utf-8")
        return _normalize_dataset_root(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _normalize_dataset_root(root: Path) -> Path:
    children = [path for path in root.iterdir() if path.is_dir()]
    root_has_images = any(
        path.is_file() and path.suffix.lower() in training.IMAGE_EXTENSIONS
        for path in root.iterdir()
    )
    if not root_has_images and len(children) == 1:
        nested = children[0]
        if any(path.is_dir() for path in nested.iterdir()):
            return nested
    return root


def dataset_source_controls(key: str, label: str) -> Path | None:
    source = st.radio(
        f"{label} 데이터 위치",
        ["ZIP 폴더 업로드 (Cloud/로컬)", "로컬 폴더 경로 (로컬 실행 전용)"],
        horizontal=True,
        key=f"{key}-source",
    )
    root: Path | None = None
    reuse_saved_root = source.startswith("ZIP")
    if source.startswith("ZIP"):
        archive = st.file_uploader(
            f"{label} 데이터 ZIP",
            type=["zip"],
            accept_multiple_files=False,
            max_upload_size=500,
            key=f"{key}-archive",
            help=(
                "학습용은 ZIP 내부 최상위가 클래스 폴더들이거나 하나의 "
                "루트 폴더 아래 클래스 폴더들이어야 합니다."
            ),
        )
        if archive is not None:
            try:
                root = _extract_dataset_archive(
                    archive.getvalue(), archive.name, key
                )
                st.success(f"압축 해제 완료: `{root}`")
            except (
                OSError,
                RuntimeError,
                NotImplementedError,
                ValueError,
                zipfile.BadZipFile,
                zipfile.LargeZipFile,
            ) as error:
                st.error(str(error))
    else:
        folder = st.text_input(
            f"{label} 폴더 경로",
            value=st.session_state.get(f"{key}-path", ""),
            placeholder=r"예: D:\datasets\train",
            key=f"{key}-path-input",
            help=(
                "Streamlit을 실행하는 컴퓨터에서 접근 가능한 절대 경로를 입력하세요. "
                "Streamlit Cloud에서는 사용자 PC 경로에 접근할 수 없으므로 ZIP을 사용하세요."
            ),
        )
        if folder.strip():
            reuse_saved_root = False
            candidate = Path(folder.strip()).expanduser()
            try:
                candidate = candidate.resolve(strict=True)
            except OSError as error:
                st.error(f"폴더 경로를 열 수 없습니다: {error}")
            else:
                if candidate.is_dir():
                    root = candidate
                    st.session_state[f"{key}-path"] = str(candidate)
                else:
                    st.error(f"폴더가 아닙니다: `{candidate}`")
        else:
            reuse_saved_root = True

    if root is not None:
        st.session_state[f"{key}-root"] = str(root)
    else:
        saved_root = st.session_state.get(f"{key}-root")
        if reuse_saved_root and saved_root:
            candidate = Path(saved_root)
            if candidate.is_dir():
                root = candidate
    return root


def _dataset_paths(root: Path, classes: list[str]) -> list[Path]:
    return [
        image
        for class_name in classes
        for image in training.image_files(root / class_name, recursive=True)
    ]


@st.cache_data(show_spinner=False)
def _cached_file_sha256(path: str, size: int, modified_ns: int) -> str:
    return training.file_sha256(Path(path))


@st.cache_data(show_spinner=False)
def _cached_checkpoint_metadata(
    path: str, size: int, modified_ns: int
) -> tuple[str | None, dict[str, str], str]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"체크포인트 메타데이터가 올바르지 않습니다: {path}")
    training_root = checkpoint.get("training_root")
    signatures = checkpoint.get("training_signatures", {})
    model_type = checkpoint.get("model_type", "segmentation")
    if (
        not isinstance(signatures, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in signatures.items()
        )
        or model_type not in ("classification", "segmentation")
    ):
        raise ValueError(f"체크포인트 학습 정보가 올바르지 않습니다: {path}")
    return (
        training_root if isinstance(training_root, str) else None,
        signatures,
        model_type,
    )


def _file_digest(path: Path) -> str:
    stat_result = path.stat()
    return _cached_file_sha256(
        str(path), stat_result.st_size, stat_result.st_mtime_ns
    )


def _mask_root(root: Path, kind: str) -> Path:
    root_hash = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:16]
    path = workspace_dir() / "masks" / root_hash / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".png", delete=False
        ) as file:
            temporary = file.name
        Image.fromarray(np.uint8(mask) * 255, mode="L").save(
            temporary, format="PNG"
        )
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _combine_canvas_mask(
    canvas_data: np.ndarray | None,
    saved_mask: np.ndarray,
    image_size: tuple[int, int],
    keep_existing: bool,
) -> np.ndarray:
    updated_mask = saved_mask.copy() if keep_existing else np.zeros_like(saved_mask)
    if canvas_data is None or canvas_data.ndim != 3 or canvas_data.shape[2] < 4:
        return updated_mask

    pixels = np.asarray(canvas_data, dtype=np.uint8)
    alpha = pixels[..., 3] > 0
    red_strokes = (
        alpha
        & (pixels[..., 0] > pixels[..., 1] * 1.2)
        & (pixels[..., 0] > pixels[..., 2] * 1.2)
    )
    blue_strokes = (
        alpha
        & (pixels[..., 2] > pixels[..., 0] * 1.15)
        & (pixels[..., 2] > pixels[..., 1] * 1.05)
    )
    add_mask = Image.fromarray(np.uint8(red_strokes) * 255).resize(
        image_size, Image.Resampling.NEAREST
    )
    erase_mask = Image.fromarray(np.uint8(blue_strokes) * 255).resize(
        image_size, Image.Resampling.NEAREST
    )
    updated_mask |= np.asarray(add_mask) > 0
    updated_mask &= ~(np.asarray(erase_mask) > 0)
    return updated_mask


def render_data_labeling() -> None:
    st.markdown("### 데이터 폴더 및 결함 라벨링")
    st.markdown(
        "데이터는 `train/클래스명/이미지` 구조로 준비하세요. "
        "Cloud에서는 ZIP으로 올리고, 로컬 실행에서는 폴더 경로도 지정할 수 있습니다."
    )
    root = dataset_source_controls("training", "학습")
    if root is None:
        st.info("학습 데이터 ZIP을 올리거나 로컬 학습 폴더 경로를 지정하세요.")
        return

    try:
        root, classes, counts = training.discover_training_data(str(root))
    except (OSError, ValueError, NotADirectoryError) as error:
        st.error(str(error))
        return

    summary = pd.DataFrame(
        [{"클래스": name, "이미지 수": count} for name, count in counts.items()]
    )
    st.dataframe(summary, width="stretch", hide_index=True)
    paths = _dataset_paths(root, classes)
    manual_root = _mask_root(root, "manual")
    batch_root = _mask_root(root, "full")
    checkpoints = [
        (training.OUTPUT_DIR / MODEL_FILES["manual"], "manual"),
        (training.OUTPUT_DIR / MODEL_FILES["full_image"], "full"),
        (training.OUTPUT_DIR / MODEL_FILES["classification"], "classification"),
    ]
    checkpoints.extend(
        (Path(st.session_state[key]), kind)
        for key, kind in (
            ("trained-manual-model", "manual"),
            ("trained-full-model", "full"),
            ("trained-classification-model", "classification"),
        )
        if st.session_state.get(key)
    )
    current_root = str(root.resolve()).casefold()
    trained_signatures: list[tuple[str, dict[str, str], str]] = []
    seen_checkpoints: set[Path] = set()
    for checkpoint_path, checkpoint_kind in checkpoints:
        if not checkpoint_path.is_file():
            continue
        resolved_checkpoint = checkpoint_path.resolve()
        if resolved_checkpoint in seen_checkpoints:
            continue
        seen_checkpoints.add(resolved_checkpoint)
        stat_result = checkpoint_path.stat()
        checkpoint_root, signatures, checkpoint_type = _cached_checkpoint_metadata(
            str(checkpoint_path), stat_result.st_size, stat_result.st_mtime_ns
        )
        if checkpoint_root and checkpoint_root.casefold() == current_root:
            trained_signatures.append(
                (checkpoint_kind, signatures, checkpoint_type)
            )

    trained_paths_by_kind = {
        "manual": set[str](),
        "full": set[str](),
        "classification": set[str](),
    }
    if trained_signatures:
        for image_path in paths:
            relative = image_path.relative_to(root).as_posix()
            image_signature = _file_digest(image_path)
            for checkpoint_kind, signatures, checkpoint_type in trained_signatures:
                if relative in trained_paths_by_kind[checkpoint_kind]:
                    continue
                if checkpoint_type == "classification":
                    current_signature = image_signature
                else:
                    mask_base = manual_root if checkpoint_kind == "manual" else batch_root
                    mask_path = training.mask_path_for(image_path, root, mask_base)
                    if not mask_path.is_file():
                        continue
                    current_signature = (
                        f"{image_signature}:{_file_digest(mask_path)}"
                    )
                if signatures.get(relative) == current_signature:
                    trained_paths_by_kind[checkpoint_kind].add(relative)

    rows = []
    for image_path in paths:
        relative = image_path.relative_to(root)
        mask_path = training.mask_path_for(image_path, root, manual_root)
        rows.append(
            {
                "클래스": relative.parts[0],
                "이미지": relative.as_posix(),
                "수동 마스크": "완료" if mask_path.is_file() else "미완료",
                "전체 마스크": (
                    "완료"
                    if training.mask_path_for(image_path, root, batch_root).is_file()
                    else "미생성"
                ),
                "수동 분할 학습": (
                    "완료"
                    if relative.as_posix() in trained_paths_by_kind["manual"]
                    else "미반영/수정됨"
                ),
                "전체 마스크 분할 학습": (
                    "완료"
                    if relative.as_posix() in trained_paths_by_kind["full"]
                    else "미반영/수정됨"
                ),
                "분류 학습": (
                    "완료"
                    if relative.as_posix() in trained_paths_by_kind["classification"]
                    else "미반영/수정됨"
                ),
            }
        )
    table = pd.DataFrame(rows)
    complete = int((table["수동 마스크"] == "완료").sum())
    st.caption(
        f"이미지 {len(paths)}장 · 수동 마스크 {complete}/{len(paths)}장 · "
        f"학습 반영(수동/전체/분류) "
        f"{len(trained_paths_by_kind['manual'])}/{len(paths)} · "
        f"{len(trained_paths_by_kind['full'])}/{len(paths)} · "
        f"{len(trained_paths_by_kind['classification'])}/{len(paths)}장 · "
        f"클래스 {len(classes)}종"
    )
    st.dataframe(table, width="stretch", hide_index=True, height=300)

    labels = [path.relative_to(root).as_posix() for path in paths]
    if not labels:
        st.warning("지원되는 이미지가 없습니다.")
        return

    selected_label = st.selectbox("라벨링할 이미지", labels, key="label-selected-image")
    selected_path = root / Path(selected_label)
    try:
        image = training.load_rgb_image(selected_path)
    except (OSError, ValueError) as error:
        st.error(f"이미지를 열 수 없습니다: {error}")
        return
    mask_path = training.mask_path_for(selected_path, root, manual_root)
    saved_mask = np.zeros((image.height, image.width), dtype=bool)
    if mask_path.is_file():
        try:
            with Image.open(mask_path) as existing:
                saved_mask = (
                    np.asarray(
                        existing.convert("L").resize(
                            image.size, Image.Resampling.NEAREST
                        ),
                        dtype=np.uint8,
                    )
                    > 127
                )
        except (OSError, ValueError) as error:
            st.error(f"저장된 마스크를 열 수 없습니다: {error}")
            return

    canvas_image = ImageOps.contain(image, (560, 520), method=Image.Resampling.LANCZOS)
    canvas_width, canvas_height = canvas_image.size
    displayed_mask = Image.fromarray(np.uint8(saved_mask) * 255).resize(
        canvas_image.size, Image.Resampling.NEAREST
    )
    preview_overlay = image.resize(canvas_image.size, Image.Resampling.LANCZOS).convert(
        "RGBA"
    )
    red_layer = Image.new("RGBA", canvas_image.size, (255, 20, 20, 0))
    red_layer.putalpha(displayed_mask.point(lambda value: 115 if value else 0))
    preview_overlay = Image.alpha_composite(preview_overlay, red_layer)

    st.caption(
        "기존 마스크를 원본 위에 표시했습니다. 아래 캔버스에서 빨간색으로 결함을 "
        "칠하고 파란색으로 지울 수 있습니다."
    )
    st.image(preview_overlay, caption=f"원본 + 저장된 라벨 · {selected_label}")
    mode = st.radio(
        "캔버스 도구",
        ["결함 칠하기", "지우개"],
        horizontal=True,
        key="label-drawing-mode",
    )
    stroke_color = "#ff1919" if mode == "결함 칠하기" else "#159dff"
    drawing = st_canvas(
        fill_color="rgba(255, 25, 25, 0.35)",
        stroke_width=st.slider(
            "브러시 크기",
            min_value=1,
            max_value=80,
            value=max(3, min(24, round(canvas_width / 32))),
            key="label-brush-size",
        ),
        stroke_color=stroke_color,
        background_color="rgba(0, 0, 0, 0)",
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="freedraw",
        key=f"label-canvas-{hashlib.sha256(selected_label.encode()).hexdigest()[:12]}",
    )
    st.caption(
        "투명 캔버스에 그린 선만 저장됩니다. 기존 마스크 유지 여부는 저장 옵션에서 선택하세요."
    )

    keep_existing = st.checkbox(
        "기존 마스크에 추가/지우기 적용",
        value=True,
        key=f"keep-mask-{hashlib.sha256(selected_label.encode()).hexdigest()[:12]}",
    )
    if st.button("현재 결함 라벨 저장", type="primary", key="save-image-mask"):
        updated_mask = _combine_canvas_mask(
            drawing.image_data,
            saved_mask,
            image.size,
            keep_existing,
        )

        if (
            not np.any(updated_mask)
            and selected_path.relative_to(root).parts[0].casefold()
            not in MASK_CLASS_NAMES
        ):
            st.error(
                "결함 이미지 라벨은 비어 있을 수 없습니다. "
                "정상 이미지는 good/normal 클래스에 두세요."
            )
        else:
            _save_mask(mask_path, updated_mask)
            st.success(
                f"마스크 저장 완료: `{mask_path}` · "
                f"결함 픽셀 {float(updated_mask.mean()):.1%}"
            )
            st.rerun()

    st.markdown("#### 전체 이미지 자동 라벨 생성")
    st.warning(
        "이 작업은 폴더의 모든 이미지를 전체 픽셀 결함 마스크(흰색)로 지정합니다. "
        "실제 결함 경계 라벨을 만드는 기능은 아니므로 이 라벨 정의에 맞는 데이터에만 사용하세요."
    )
    confirm_batch = st.checkbox(
        "모든 이미지 전체를 결함 영역으로 지정하는 것을 확인합니다.",
        key="confirm-full-image-labels",
    )
    if st.button(
        "전체 이미지 라벨 및 마스크 생성",
        disabled=not confirm_batch,
        key="generate-full-image-labels",
    ):
        records: list[dict[str, str]] = []
        for image_path in paths:
            try:
                source = training.load_rgb_image(image_path)
            except (OSError, ValueError) as error:
                st.error(f"이미지를 읽을 수 없습니다: {image_path} · {error}")
                continue
            full_mask_path = training.mask_path_for(image_path, root, batch_root)
            _save_mask(
                full_mask_path,
                np.ones((source.height, source.width), dtype=bool),
            )
            records.append(
                {
                    "이미지": image_path.relative_to(root).as_posix(),
                    "분류 라벨": image_path.relative_to(root).parts[0],
                    "결함 마스크": image_path.relative_to(root)
                    .with_suffix(".mask.png")
                    .as_posix(),
                }
            )
        if records:
            labels_csv = pd.DataFrame(records).to_csv(index=False).encode("utf-8-sig")
            st.session_state["batch-label-csv"] = labels_csv
            st.success(f"전체 마스크를 {len(records)}장 생성했습니다. 수동 마스크와 분리 저장했습니다.")
            st.rerun()

    labels_csv = st.session_state.get("batch-label-csv")
    if labels_csv:
        st.download_button(
            "전체 이미지 분류 라벨 CSV 다운로드",
            data=labels_csv,
            file_name="image_labels.csv",
            mime="text/csv",
        )

    full_masks_archive = st.session_state.get("full-mask-archive")
    if full_masks_archive:
        st.download_button(
            "전체 이미지 마스크 ZIP 다운로드",
            data=full_masks_archive,
            file_name="full_image_masks.zip",
            mime="application/zip",
        )
    if st.button("전체 이미지 마스크를 ZIP으로 준비", key="archive-full-image-masks"):
        st.session_state["full-mask-archive"] = _mask_archive(paths, root, batch_root)
        st.rerun()

    masks_archive = st.session_state.get("manual-mask-archive")
    if masks_archive:
        st.download_button(
            "수동 마스크 폴더 다운로드",
            data=masks_archive,
            file_name="manual_masks.zip",
            mime="application/zip",
        )
    if st.button("현재 수동 마스크를 ZIP으로 준비", key="archive-manual-masks"):
        st.session_state["manual-mask-archive"] = _mask_archive(
            paths, root, manual_root
        )
        st.rerun()


def _mask_archive(paths: list[Path], root: Path, mask_root: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            mask_path = training.mask_path_for(path, root, mask_root)
            if mask_path.is_file():
                archive.write(
                    mask_path,
                    arcname=path.relative_to(root).with_suffix(".mask.png").as_posix(),
                )
    return buffer.getvalue()


def _train_and_render(
    generator: Any, epochs: int, model_path: Path, model_key: str
) -> None:
    progress = st.progress(0, text="학습 준비 중...")
    status = st.empty()
    history_view = st.empty()
    last_status = ""
    last_history = pd.DataFrame()
    for last_status, last_history in generator:
        status.markdown(last_status)
        if not last_history.empty:
            history_view.dataframe(
                last_history, width="stretch", hide_index=True
            )
            progress.progress(
                min(1.0, len(last_history) / epochs),
                text=f"Epoch {len(last_history)}/{epochs}",
            )
    if not model_path.is_file():
        raise FileNotFoundError(f"학습은 종료됐지만 모델 파일을 찾을 수 없습니다: {model_path}")
    st.session_state[model_key] = str(model_path)
    stat_result = model_path.stat()
    st.session_state[f"{model_key}-training-metadata"] = (
        _cached_checkpoint_metadata(
            str(model_path), stat_result.st_size, stat_result.st_mtime_ns
        )
    )
    status.success(last_status)
    st.download_button(
        "학습한 모델 체크포인트 다운로드",
        data=model_path.read_bytes(),
        file_name=model_path.name,
        mime="application/octet-stream",
        key=f"download-{model_key}",
    )


def render_training() -> None:
    st.markdown("### 별도 학습 작업")
    root_value = st.session_state.get("training-root")
    if not root_value or not Path(root_value).is_dir():
        st.info("먼저 ‘데이터 라벨링’ 탭에서 학습 ZIP 또는 로컬 폴더를 선택하세요.")
        return
    root = Path(root_value)
    try:
        root, classes, counts = training.discover_training_data(str(root))
    except (OSError, ValueError, NotADirectoryError) as error:
        st.error(str(error))
        return
    st.caption(
        f"데이터: `{root}` · {len(classes)}개 클래스 · "
        + ", ".join(f"{name}: {count}장" for name, count in counts.items())
    )

    with st.form("training-parameters"):
        training_mode = st.selectbox(
            "학습할 모델",
            [
                "수동 마스크 기반 분할",
                "전체 이미지 마스크 기반 분할",
                "마스크 없는 이미지 분류",
            ],
        )
        first, second, third = st.columns(3)
        epochs = first.number_input("Epoch", 1, 100, 15, 1)
        batch_size = second.selectbox("Batch size", [1, 2, 4, 8, 16], index=2)
        learning_rate = third.number_input(
            "Learning rate", min_value=0.00001, max_value=0.1, value=0.001,
            format="%.5f",
        )
        start_training = st.form_submit_button("학습 시작", type="primary")

    if not start_training:
        return
    try:
        models_dir = workspace_dir() / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        if training_mode == "마스크 없는 이미지 분류":
            model_path = models_dir / "image_classifier.pt"
            generator = training.train_classification_model(
                str(root), int(epochs), int(batch_size), float(learning_rate),
                model_path=model_path,
            )
            model_key = "trained-classification-model"
        else:
            kind = (
                "manual"
                if training_mode == "수동 마스크 기반 분할"
                else "full"
            )
            mask_root = _mask_root(root, kind)
            model_path = models_dir / (
                "segmentation_classifier.pt"
                if kind == "manual"
                else "full_image_segmentation.pt"
            )
            generator = training.train_model(
                str(root),
                int(epochs),
                int(batch_size),
                float(learning_rate),
                mask_root=mask_root,
                model_path=model_path,
                training_name=training_mode,
            )
            model_key = (
                "trained-manual-model"
                if kind == "manual"
                else "trained-full-model"
            )
        _train_and_render(generator, int(epochs), model_path, model_key)
    except (OSError, ValueError, RuntimeError) as error:
        st.error(f"학습에 실패했습니다: {error}")
        st.info(
            "분류 클래스마다 학습/검증용 이미지가 최소 2장 필요합니다. "
            "분할 학습에는 해당 유형의 마스크가 모든 이미지에 필요합니다."
        )


def _model_choices() -> dict[str, tuple[str, str]]:
    choices: dict[str, tuple[str, str]] = {}
    for mode, filename in MODEL_FILES.items():
        if (training.OUTPUT_DIR / filename).is_file():
            choices[f"저장소: {filename}"] = ("bundled", mode)
    for state_key, label in (
        ("trained-manual-model", "이번 세션 학습: 수동 분할"),
        ("trained-full-model", "이번 세션 학습: 전체 마스크 분할"),
        ("trained-classification-model", "이번 세션 학습: 분류 전용"),
    ):
        if st.session_state.get(state_key):
            choices[label] = ("trained", state_key)
    return choices


def _load_test_model(
    choice: str,
    choices: dict[str, tuple[str, str]],
    uploaded: Any = None,
) -> tuple[Any, list[str], int, str, str]:
    if choice == "체크포인트 직접 업로드":
        raw = (
            uploaded.getvalue()
            if uploaded is not None
            else st.session_state.get("folder-test-checkpoint")
        )
        if raw is None:
            raise ValueError("테스트 전에 .pt/.pth 체크포인트를 올리세요.")
        model_name = (
            uploaded.name
            if uploaded is not None
            else st.session_state.get("folder-test-checkpoint-name", "uploaded.pt")
        )
        model, classes, image_size, kind = load_model_from_bytes(raw, model_name)
        return model, classes, image_size, kind, model_name

    source, value = choices[choice]
    if source == "bundled":
        if value not in MODEL_FILES:
            raise ValueError(f"알 수 없는 기본 모델 유형입니다: {value}")
        mode = cast(ModelMode, value)
        model, classes, image_size = load_model(training.OUTPUT_DIR, mode)
        kind = "classification" if mode == "classification" else "segmentation"
        return model, classes, image_size, kind, MODEL_FILES[mode]

    model_path = Path(st.session_state[value])
    model, classes, image_size, kind = load_model_from_bytes(
        model_path.read_bytes(), model_path.name
    )
    return model, classes, image_size, kind, model_path.name


def _run_folder_inference(
    root: Path,
    model: Any,
    classes: list[str],
    image_size: int,
    model_type: str,
    progress: Any,
) -> tuple[pd.DataFrame, bytes]:
    paths = training.image_files(root, recursive=True)
    if not paths:
        raise ValueError("선택한 폴더와 하위 폴더에서 지원되는 이미지가 없습니다.")
    if len(paths) > MAX_TEST_IMAGES:
        raise ValueError(f"이미지가 {len(paths):,}장입니다. 최대 {MAX_TEST_IMAGES:,}장까지 처리합니다.")

    rows: list[dict[str, Any]] = []
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(
        archive_buffer, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for index, path in enumerate(paths, start=1):
            try:
                image = training.load_rgb_image(path)
            except (OSError, ValueError) as error:
                rows.append(
                    {
                        "이미지": path.relative_to(root).as_posix(),
                        "상태": f"읽기 실패: {error}",
                        "분류 결과": "—",
                        "분류 점수": "—",
                        "결함 면적": "—",
                    }
                )
                continue

            predicted, confidence, mask = predict_image(
                image, model, classes, image_size
            )
            relative = path.relative_to(root)
            mask_filename = (
                relative.with_suffix(".mask.png").as_posix()
                if model_type == "segmentation" and mask is not None
                else None
            )
            rows.append(
                {
                    "이미지": relative.as_posix(),
                    "상태": "완료",
                    "분류 결과": predicted,
                    "분류 점수": f"{confidence:.1%}",
                    "결함 면적": f"{float(mask.mean()):.1%}" if mask is not None else "미실행",
                    "예측 마스크": mask_filename or "—",
                }
            )
            if mask is not None and mask_filename is not None:
                mask_file = io.BytesIO()
                Image.fromarray(np.uint8(mask) * 255, mode="L").save(
                    mask_file, format="PNG"
                )
                archive.writestr(mask_filename, mask_file.getvalue())
            if index % 25 == 0 or index == len(paths):
                progress.progress(
                    index / len(paths),
                    text=f"이미지 분석 중: {index}/{len(paths)}",
                )

    frame = pd.DataFrame(rows)
    return frame, archive_buffer.getvalue()


def render_folder_test() -> None:
    st.markdown("### 테스트 폴더 전체 분석")
    st.markdown(
        "하위 폴더까지 이미지 전체를 분석합니다. 결과는 CSV와 예측 마스크 ZIP으로 내려받을 수 있습니다."
    )
    root = dataset_source_controls("test", "테스트")
    choices = _model_choices()
    options = list(choices) + ["체크포인트 직접 업로드"]
    if not options:
        st.error("저장된 기본 모델이 없습니다. 학습을 완료하거나 체크포인트를 업로드하세요.")
        return
    selected = st.selectbox("테스트에 사용할 모델", options, key="folder-test-model")
    checkpoint = None
    if selected == "체크포인트 직접 업로드":
        checkpoint = st.file_uploader(
            "테스트용 모델 파일",
            type=["pt", "pth"],
            max_upload_size=20,
            key="folder-test-checkpoint",
        )
    if root is None:
        st.info("테스트 폴더 ZIP을 올리거나 로컬 테스트 폴더 경로를 지정하세요.")
        return
    if st.button("폴더 전체 이미지 분석", type="primary", key="run-folder-test"):
        try:
            model, classes, image_size, model_type, model_name = _load_test_model(
                selected, choices, checkpoint
            )
            if checkpoint is not None:
                st.session_state["folder-test-checkpoint"] = checkpoint.getvalue()
                st.session_state["folder-test-checkpoint-name"] = checkpoint.name
            progress = st.progress(0, text="모델을 불러왔습니다. 폴더 이미지를 검색합니다.")
            frame, masks_zip = _run_folder_inference(
                root, model, classes, image_size, model_type, progress
            )
            progress.progress(1.0, text="폴더 전체 분석이 완료됐습니다.")
            st.session_state["folder-test-results"] = frame
            st.session_state["folder-test-masks"] = masks_zip
            st.session_state["folder-test-model-name"] = model_name
            st.session_state["folder-test-model-choice"] = selected
            st.session_state["folder-test-root"] = str(root)
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"폴더 분석을 완료하지 못했습니다: {error}")

    frame = st.session_state.get("folder-test-results")
    if frame is None:
        return
    st.caption(f"모델: {st.session_state['folder-test-model-name']} · 분석 행: {len(frame)}장")
    successful = frame[frame["상태"] == "완료"] if "상태" in frame else frame
    st.metric("정상 분석된 이미지", f"{len(successful)} / {len(frame)}")
    st.dataframe(frame, width="stretch", hide_index=True)
    left, right = st.columns(2)
    left.download_button(
        "폴더 분석 결과 CSV 다운로드",
        data=frame.to_csv(index=False).encode("utf-8-sig"),
        file_name="folder_predictions.csv",
        mime="text/csv",
        width="stretch",
    )
    right.download_button(
        "예측 마스크 ZIP 다운로드",
        data=st.session_state["folder-test-masks"],
        file_name="folder_predicted_masks.zip",
        mime="application/zip",
        width="stretch",
        disabled=not st.session_state["folder-test-masks"],
    )
    st.markdown("#### 분석 이미지 미리보기")
    preview_names = successful["이미지"].tolist()
    if preview_names:
        selected_image = st.selectbox(
            "표시할 분석 이미지",
            preview_names,
            key="folder-test-preview-selection",
        )
        preview_root = Path(st.session_state["folder-test-root"]).resolve()
        preview_path = (preview_root / Path(selected_image)).resolve()
        if not preview_path.is_relative_to(preview_root):
            st.error("선택된 이미지 경로가 테스트 폴더 바깥을 가리킵니다.")
            return
        try:
            preview_model, preview_classes, preview_size, preview_type, _ = (
                _load_test_model(
                    st.session_state["folder-test-model-choice"],
                    choices,
                )
            )
            image = training.load_rgb_image(preview_path)
            predicted, confidence, mask = predict_image(
                image, preview_model, preview_classes, preview_size
            )
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"선택 이미지 미리보기를 만들지 못했습니다: {error}")
            return
        st.caption(
            f"{selected_image} · {predicted} · {confidence:.1%} · "
            f"{'분할' if preview_type == 'segmentation' else '분류 전용'}"
        )
        if mask is None:
            st.image(image, caption="원본 이미지 · 분류 전용 모델", width="stretch")
        else:
            first, second = st.columns(2)
            first.image(image, caption="원본", width="stretch")
            second.image(
                render_mask_overlay(image, mask),
                caption=f"결함 예측 영역 · {float(mask.mean()):.1%}",
                width="stretch",
            )
