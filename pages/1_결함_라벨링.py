import sys
from pathlib import Path

import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from streamlit_workflows import render_data_labeling


st.set_page_config(
    page_title="NEU VISION | 결함 라벨링",
    page_icon="🖌️",
    layout="wide",
)
st.title("결함 라벨링 및 데이터 준비")
render_data_labeling()
