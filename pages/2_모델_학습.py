import sys
from pathlib import Path

import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from streamlit_workflows import render_training


st.set_page_config(
    page_title="NEU VISION | 모델 학습",
    page_icon="🧠",
    layout="wide",
)
st.title("모델 학습")
render_training()
