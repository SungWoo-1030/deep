from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image

from model_runtime import (
    IMAGE_SIZE,
    MODEL_FILES,
    ModelMode,
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
      <p>학습된 모델을 선택하고 사진을 올리면 결함 분류와 영역 예측 결과를 확인할 수 있습니다.</p>
    </section>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="선택한 AI 모델을 불러오는 중...")
def get_model(mode: ModelMode) -> tuple[Any, list[str], int]:
    return load_model(OUTPUT_DIR, mode)


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
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, upload in enumerate(uploads, start=1):
        try:
            image = open_uploaded_image(upload.getvalue())
        except ValueError as error:
            errors.append(f"{upload.name}: {error}")
            continue

        predicted_class, confidence, mask = predict_image(
            image, model, classes, image_size
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
    selected_label = st.selectbox("사용할 학습 모델", list(MODEL_OPTIONS))
    selected_mode = MODEL_OPTIONS[selected_label]
    st.caption(f"모델 파일: `outputs/{MODEL_FILES[selected_mode]}`")
    st.caption(f"입력 크기: {IMAGE_SIZE} × {IMAGE_SIZE} 픽셀")

    st.markdown("---")
    st.markdown("**모델 안내**")
    if selected_mode == "manual":
        st.caption("브러시로 직접 표시한 마스크와 클래스 라벨을 학습한 모델입니다.")
    elif selected_mode == "full_image":
        st.caption("전체 이미지 마스크와 클래스 라벨로 별도 학습한 모델입니다.")
    else:
        st.caption("클래스 분류만 수행하며 결함 영역 마스크는 생성하지 않습니다.")

try:
    model, class_names, input_size = get_model(selected_mode)
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.info(
        "GitHub 저장소에 `outputs` 폴더와 선택한 모델 파일을 포함했는지 확인하세요."
    )
    st.stop()

uploads = st.file_uploader(
    "검사할 이미지 업로드",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
    help="여러 장을 한 번에 선택할 수 있습니다. 업로드 파일은 서버 디스크에 저장하지 않습니다.",
)

run_inference = st.button(
    "선택 이미지 분석",
    type="primary",
    disabled=not uploads,
    width="stretch",
)

if run_inference:
    try:
        with st.spinner(f"{len(uploads)}장 이미지를 분석하고 있습니다..."):
            st.session_state["inspection_results"] = process_uploads(
                uploads, model, class_names, input_size
            )
            st.session_state["inspection_mode"] = selected_mode
    except ValueError as error:
        st.error(str(error))
        st.stop()

results = st.session_state.get("inspection_results", [])
results_mode = st.session_state.get("inspection_mode")
if results and results_mode == selected_mode:
    average_confidence = sum(item["confidence"] for item in results) / len(results)
    st.markdown("### 검사 요약")
    metric_columns = st.columns(3)
    metric_columns[0].metric("분석 이미지", f"{len(results)}장")
    metric_columns[1].metric("학습 클래스", f"{len(class_names)}종")
    metric_columns[2].metric("평균 분류 점수", f"{average_confidence:.1%}")

    result_rows = [
        {
            "이미지": item["filename"],
            "분류 결과": item["predicted_class"],
            "분류 점수 (보정되지 않음)": f"{item['confidence']:.1%}",
            "결함 표시 면적": f"{item['area']:.1%}" if item["area"] is not None else "미실행",
        }
        for item in results
    ]
    st.dataframe(pd.DataFrame(result_rows), width="stretch", hide_index=True)

    csv_data = pd.DataFrame(result_rows).to_csv(index=False).encode("utf-8-sig")
    download_columns = st.columns(2)
    download_columns[0].download_button(
        "분류 결과 CSV 다운로드",
        data=csv_data,
        file_name="inspection_results.csv",
        mime="text/csv",
        width="stretch",
    )
    if selected_mode != "classification":
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
