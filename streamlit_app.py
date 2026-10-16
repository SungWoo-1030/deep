from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from model_runtime import (
    IMAGE_SIZE,
    MODEL_FILES,
    ModelMode,
    load_model_from_bytes,
    open_uploaded_image,
    load_model,
    predict_image,
    render_mask_overlay,
)


APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"
SUPPORTED_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]
MODEL_OPTIONS: dict[str, ModelMode] = {
    "수동 마스크 기반 분할 모델": "manual",
    "전체 이미지 마스크 기반 분할 모델": "full_image",
    "이미지 분류 전용 모델 (마스크 없음)": "classification",
}

st.set_page_config(
    page_title="NEU VISION | AI 검사 스튜디오",
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(0, 217, 255, .08), transparent 32%),
            #07111f;
        color: #e6f1ff;
    }
    [data-testid="stHeader"] { background: rgba(7, 17, 31, .78); }
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background: #0b192a;
        border-right: 1px solid rgba(92, 137, 168, .22);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] label {
        color: #d2e1ef !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f2f8ff !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        border-color: #2a425d;
        background: #12243a;
        color: #e6f1ff;
    }
    [data-testid="stMetric"] {
        padding: 16px 18px;
        border: 1px solid rgba(92, 137, 168, .28);
        border-radius: 14px;
        background: rgba(13, 27, 44, .84);
    }
    [data-testid="stFileUploader"] section {
        border: 1px dashed #397089;
        background: rgba(12, 29, 47, .76);
    }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span {
        color: #d2e1ef !important;
    }
    [data-testid="stFileUploader"] button {
        border: 1px solid #397089;
        background: #123047;
        color: #e6f1ff;
    }
    .stButton > button:disabled {
        border-color: #2a425d;
        background: #14263a;
        color: #aebfd1;
        opacity: 1;
    }
    [data-testid="stExpander"] {
        border: 1px solid rgba(92, 137, 168, .24);
        border-radius: 12px;
        background: rgba(13, 27, 44, .72);
    }
    .hero {
        margin: 8px 0 24px;
        padding: 32px;
        border: 1px solid rgba(86, 160, 198, .28);
        border-radius: 22px;
        background: linear-gradient(120deg, rgba(13, 30, 49, .98), rgba(10, 20, 39, .94));
    }
    .hero-kicker {
        color: #62e5ff;
        font-family: ui-monospace, Consolas, monospace;
        font-size: 12px;
        letter-spacing: .16em;
    }
    .hero h1 {
        margin: 12px 0 8px;
        color: #f2f8ff;
        font-size: clamp(30px, 4vw, 46px);
        letter-spacing: -.04em;
    }
    .hero p { color: #bfd0e2; font-size: 16px; }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"] { color: #d2e1ef; }
    code {
        color: #a5efff !important;
        background: #162a40 !important;
    }
    </style>
    <section class="hero">
      <div class="hero-kicker">NEU / VISION SYSTEMS · AI INSPECTION STUDIO</div>
      <h1>이미지로 시작하는<br>스마트 품질 검사</h1>
      <p>결함 라벨링부터 모델 학습, 단일 이미지 및 폴더 전체 검사까지 수행할 수 있습니다.</p>
    </section>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="선택한 AI 모델을 불러오는 중...")
def get_model(mode: ModelMode) -> tuple[Any, list[str], int]:
    return load_model(OUTPUT_DIR, mode)


@st.cache_resource(show_spinner="업로드한 AI 모델을 안전하게 불러오는 중...")
def get_uploaded_model(
    checkpoint_data: bytes, model_name: str
) -> tuple[Any, list[str], int, str]:
    return load_model_from_bytes(checkpoint_data, model_name)


def upload_key(filename: str, *, is_mask: bool = False) -> str:
    name = Path(filename.replace("\\", "/")).stem.casefold()
    if is_mask and name.endswith(".mask"):
        name = name[:-5]
    return name


def load_reference_masks(
    uploads: list[Any],
) -> tuple[dict[str, tuple[np.ndarray, str]], list[str]]:
    reference_masks: dict[str, tuple[np.ndarray, str]] = {}
    errors: list[str] = []
    ambiguous_keys: set[str] = set()

    for upload in uploads:
        key = upload_key(upload.name, is_mask=True)
        try:
            with Image.open(io.BytesIO(upload.getvalue())) as source:
                mask_image = source.convert("L")
                mask = np.asarray(mask_image, dtype=np.uint8) > 127
        except (OSError, UnidentifiedImageError, ValueError) as error:
            errors.append(f"{upload.name}: 읽을 수 없는 마스크 파일입니다 ({error}).")
            continue

        if key in ambiguous_keys:
            continue
        if key in reference_masks:
            previous_name = reference_masks.pop(key)[1]
            ambiguous_keys.add(key)
            errors.append(
                f"같은 이미지 이름에 대응하는 마스크가 중복되었습니다: "
                f"{previous_name}, {upload.name}"
            )
            continue
        reference_masks[key] = (mask, upload.name)

    return reference_masks, errors


def make_mask_archive(results: list[dict[str, Any]]) -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for item in results:
            mask = item["mask"]
            if mask is None:
                continue
            image = Image.fromarray(mask.astype("uint8") * 255, mode="L")
            mask_file = io.BytesIO()
            image.save(mask_file, format="PNG")
            bundle.writestr(item["mask_filename"], mask_file.getvalue())
    return archive.getvalue()


def process_uploads(
    uploads: list[Any],
    model: Any,
    classes: list[str],
    image_size: int,
    reference_masks: dict[str, tuple[np.ndarray, str]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    image_key_counts: dict[str, int] = {}
    for upload in uploads:
        key = upload_key(upload.name)
        image_key_counts[key] = image_key_counts.get(key, 0) + 1

    for index, upload in enumerate(uploads, start=1):
        try:
            image = open_uploaded_image(upload.getvalue())
        except ValueError as error:
            errors.append(f"{upload.name}: {error}")
            continue

        predicted_class, confidence, mask = predict_image(
            image, model, classes, image_size
        )
        reference_mask = None
        reference_mask_filename = None
        iou = None
        dice = None
        key = upload_key(upload.name)
        if key in reference_masks:
            reference_mask, reference_mask_filename = reference_masks[key]
            if image_key_counts[key] > 1:
                errors.append(
                    f"{upload.name}: 이미지 파일명이 중복되어 정답 마스크와 "
                    "안전하게 연결할 수 없습니다."
                )
                reference_mask = None
            elif reference_mask.shape != (image.height, image.width):
                errors.append(
                    f"{reference_mask_filename}: 원본 이미지 {upload.name}와 "
                    "크기가 달라 비교할 수 없습니다."
                )
                reference_mask = None
            elif mask is None:
                errors.append(
                    f"{reference_mask_filename}: 분류 전용 모델은 마스크를 "
                    "예측하지 않아 비교할 수 없습니다."
                )
                reference_mask = None
            else:
                intersection = int(np.count_nonzero(mask & reference_mask))
                union = int(np.count_nonzero(mask | reference_mask))
                predicted_area = int(np.count_nonzero(mask))
                reference_area = int(np.count_nonzero(reference_mask))
                iou = intersection / union if union else 1.0
                dice_denominator = predicted_area + reference_area
                dice = (
                    2 * intersection / dice_denominator
                    if dice_denominator
                    else 1.0
                )

        safe_stem = Path(upload.name.replace("\\", "/")).stem
        results.append(
            {
                "filename": upload.name,
                "image": image,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "mask": mask,
                "area": float(mask.mean()) if mask is not None else None,
                "reference_mask": reference_mask,
                "reference_mask_filename": reference_mask_filename,
                "iou": iou,
                "dice": dice,
                "mask_filename": f"{index:04d}_{safe_stem}_mask.png",
            }
        )

    for error in errors:
        st.error(error)
    if not results and errors:
        raise ValueError("처리할 수 있는 이미지가 없습니다. 파일 형식을 확인하세요.")
    return results


with st.sidebar:
    st.markdown("### 검사 설정")
    model_source = st.radio(
        "모델 파일 선택",
        ["저장소 기본 모델", "체크포인트 업로드"],
    )
    model = None
    class_names: list[str] = []
    input_size = IMAGE_SIZE
    model_kind: str | None = None
    selected_mode: ModelMode | None = None
    model_identity: str

    if model_source == "저장소 기본 모델":
        selected_label = st.selectbox("사용할 학습 모델", list(MODEL_OPTIONS))
        selected_mode = MODEL_OPTIONS[selected_label]
        model_identity = selected_mode
        st.caption(f"모델 파일: `outputs/{MODEL_FILES[selected_mode]}`")
        try:
            model, class_names, input_size = get_model(selected_mode)
            model_kind = (
                "classification" if selected_mode == "classification" else "segmentation"
            )
        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))
            st.info("저장소의 `outputs` 폴더에 선택한 모델 파일을 포함했는지 확인하세요.")
    else:
        uploaded_checkpoint = st.file_uploader(
            "학습된 모델 체크포인트",
            type=["pt", "pth"],
            accept_multiple_files=False,
            max_upload_size=20,
            help="이 앱에서 학습해 저장한 .pt/.pth 체크포인트를 올리세요. 파일은 저장하지 않습니다.",
        )
        if uploaded_checkpoint is None:
            st.info("분석을 시작하려면 학습된 .pt 또는 .pth 모델 파일을 올리세요.")
            model_identity = "uploaded:none"
        else:
            checkpoint_data = uploaded_checkpoint.getvalue()
            model_identity = "uploaded:" + hashlib.sha256(checkpoint_data).hexdigest()
            try:
                model, class_names, input_size, model_kind = get_uploaded_model(
                    checkpoint_data, uploaded_checkpoint.name
                )
                st.caption(f"업로드한 모델: `{uploaded_checkpoint.name}`")
            except ValueError as error:
                st.error(str(error))

    if model is not None:
        st.caption(f"입력 크기: {input_size} × {input_size} 픽셀")

    st.markdown("---")
    st.markdown("**모델 안내**")
    if model_kind == "classification":
        st.caption("이미지 분류만 수행하며 결함 영역 마스크는 생성하지 않습니다.")
    elif (
        model_kind == "segmentation"
        and model_source == "저장소 기본 모델"
        and selected_mode == "manual"
    ):
        st.caption("브러시로 직접 표시한 마스크와 클래스 라벨을 학습한 모델입니다.")
    elif (
        model_kind == "segmentation"
        and model_source == "저장소 기본 모델"
        and selected_mode == "full_image"
    ):
        st.caption("전체 이미지 마스크와 클래스 라벨로 별도 학습한 모델입니다.")
    elif model_kind == "segmentation":
        st.caption("업로드한 분할 체크포인트로 이미지 분류와 결함 영역 예측을 수행합니다.")

uploads = st.file_uploader(
    "검사할 이미지 업로드",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
    help="여러 장을 한 번에 선택할 수 있습니다. 업로드 파일은 서버 디스크에 저장하지 않습니다.",
)
mask_uploads = st.file_uploader(
    "비교할 정답 마스크 업로드 (선택)",
    type=["png", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
    help=(
        "흑백 이진 마스크를 올리세요. 이미지와 파일명(확장자 제외)이 같거나 "
        "이미지명.mask.png 형식이어야 하며, 원본과 크기가 같아야 합니다."
    ),
)

run_inference = st.button(
    "선택 이미지 분석",
    type="primary",
    disabled=not uploads or model is None,
    width="stretch",
)

if run_inference and uploads and model is not None:
    try:
        references, mask_errors = load_reference_masks(mask_uploads or [])
        for error in mask_errors:
            st.error(error)
        with st.spinner(f"{len(uploads)}장 이미지를 분석하고 있습니다..."):
            st.session_state["inspection_results"] = process_uploads(
                uploads, model, class_names, input_size, references
            )
            st.session_state["inspection_mode"] = model_identity
    except ValueError as error:
        st.error(str(error))
        st.stop()

results = st.session_state.get("inspection_results", [])
results_mode = st.session_state.get("inspection_mode")
if results and results_mode == model_identity:
    average_confidence = sum(item["confidence"] for item in results) / len(results)
    compared_results = [item for item in results if item["iou"] is not None]
    st.markdown("### 검사 요약")
    metric_columns = st.columns(5)
    metric_columns[0].metric("분석 이미지", f"{len(results)}장")
    metric_columns[1].metric("학습 클래스", f"{len(class_names)}종")
    metric_columns[2].metric("평균 분류 점수", f"{average_confidence:.1%}")
    metric_columns[3].metric(
        "평균 마스크 IoU",
        f"{np.mean([item['iou'] for item in compared_results]):.3f}"
        if compared_results
        else "—",
    )
    metric_columns[4].metric(
        "평균 마스크 Dice",
        f"{np.mean([item['dice'] for item in compared_results]):.3f}"
        if compared_results
        else "—",
    )

    result_rows = [
        {
            "이미지": item["filename"],
            "분류 결과": item["predicted_class"],
            "분류 점수 (보정되지 않음)": f"{item['confidence']:.1%}",
            "결함 표시 면적": f"{item['area']:.1%}" if item["area"] is not None else "미실행",
            "정답 마스크": item["reference_mask_filename"] or "미업로드",
            "마스크 IoU": f"{item['iou']:.3f}" if item["iou"] is not None else "—",
            "마스크 Dice": f"{item['dice']:.3f}" if item["dice"] is not None else "—",
        }
        for item in results
    ]
    st.dataframe(pd.DataFrame(result_rows), width="stretch", hide_index=True)

    if mask_uploads:
        matched_masks = {
            item["reference_mask_filename"]
            for item in results
            if item["reference_mask_filename"] is not None
        }
        unmatched_masks = [
            upload.name for upload in mask_uploads if upload.name not in matched_masks
        ]
        without_masks = sum(
            item["reference_mask_filename"] is None for item in results
        )
        if unmatched_masks:
            st.warning(
                "이미지와 이름이 일치하지 않아 사용하지 않은 정답 마스크: "
                + ", ".join(unmatched_masks)
            )
        if without_masks:
            st.info(f"정답 마스크가 연결되지 않은 이미지: {without_masks}장")
    else:
        st.caption("정답 마스크를 올리지 않으면 IoU/Dice 비교는 생략됩니다.")

    csv_data = pd.DataFrame(result_rows).to_csv(index=False).encode("utf-8-sig")
    download_columns = st.columns(2)
    download_columns[0].download_button(
        "분류 결과 CSV 다운로드",
        data=csv_data,
        file_name="inspection_results.csv",
        mime="text/csv",
        width="stretch",
    )
    if model_kind == "segmentation":
        download_columns[1].download_button(
            "예측 마스크 PNG 묶음 다운로드",
            data=make_mask_archive(results),
            file_name="predicted_masks.zip",
            mime="application/zip",
            width="stretch",
        )

    st.markdown("### 이미지별 분석 결과")
    for item in results:
        title = (
            f"{item['filename']} · {item['predicted_class']} · "
            f"{item['confidence']:.1%}"
        )
        with st.expander(title):
            if item["mask"] is None:
                st.image(
                    item["image"],
                    caption="원본 이미지 · 분류 전용 모델은 결함 영역을 예측하지 않습니다.",
                    width="stretch",
                )
            elif item["reference_mask"] is not None:
                original, reference, predicted = st.columns(3)
                original.image(
                    item["image"],
                    caption="원본 이미지",
                    width="stretch",
                )
                reference.image(
                    Image.fromarray(
                        item["reference_mask"].astype("uint8") * 255, mode="L"
                    ),
                    caption=f"정답 마스크 · {item['reference_mask_filename']}",
                    width="stretch",
                )
                predicted.image(
                    render_mask_overlay(item["image"], item["mask"]),
                    caption=(
                        f"예측 마스크 · IoU {item['iou']:.3f} · "
                        f"Dice {item['dice']:.3f}"
                    ),
                    width="stretch",
                )
            else:
                left, right = st.columns(2)
                left.image(
                    item["image"],
                    caption="원본 이미지",
                    width="stretch",
                )
                right.image(
                    render_mask_overlay(item["image"], item["mask"]),
                    caption=f"결함 예측 영역 · {item['area']:.1%}",
                    width="stretch",
                )
                mask_bytes = io.BytesIO()
                Image.fromarray(item["mask"].astype("uint8") * 255, mode="L").save(
                    mask_bytes, format="PNG"
                )
                st.download_button(
                    "이 이미지의 예측 마스크 다운로드",
                    data=mask_bytes.getvalue(),
                    file_name=item["mask_filename"],
                    mime="image/png",
                    key=f"mask-{item['mask_filename']}",
                )

st.markdown("---")
st.caption(
    "분류 점수는 보정된 확률이나 정확도 보증이 아닙니다. "
    "결함 영역은 모델의 예측이며 정답을 보장하지 않습니다."
)
