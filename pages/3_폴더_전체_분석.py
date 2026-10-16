import sys
from pathlib import Path

import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from streamlit_workflows import render_folder_test


st.set_page_config(
    page_title="NEU VISION | 폴더 분석",
    page_icon="🔍",
    layout="wide",
)
st.title("테스트 폴더 전체 분석")
render_folder_test()
