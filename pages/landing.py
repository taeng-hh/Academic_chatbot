"""Landing page: Gachon University logo above the academy assistant title."""
from pathlib import Path

import streamlit as st


# Use an official local logo when provided. Otherwise, display the logo from
# Gachon University's official identity page (not a third-party mirror).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_LOGO = PROJECT_ROOT / "assets" / "gachon_logo.png"
OFFICIAL_LOGO_URL = "https://www.gachon.ac.kr/sites/kor/images/sub/slogan_2.png"

st.markdown('<div style="height: 12vh;"></div>', unsafe_allow_html=True)

left, center, right = st.columns([1, 2, 1])
with center:
    st.image(
        str(LOCAL_LOGO) if LOCAL_LOGO.is_file() else OFFICIAL_LOGO_URL,
        caption=None,
        use_container_width=True,
    )
    st.markdown(
        "<h1 style='text-align: center; margin-top: 1.5rem; "
        "margin-bottom: 2.5rem;'>가천 학원 평가 도우미</h1>",
        unsafe_allow_html=True,
    )
    if st.button("모의고사 분석 시작", type="primary", use_container_width=True):
        st.switch_page("pages/analysis.py")
