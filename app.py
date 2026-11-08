"""Streamlit entrypoint: st.navigation for the two required pages."""
from __future__ import annotations

import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
st.set_page_config(page_title="AI 모의고사 평가 도우미", page_icon="📊", layout="wide")

# Local: .env / Cloud: Streamlit Secrets (never commit the real secrets file)
try:
    if not os.getenv("OPENAI_API_KEY"):
        key = st.secrets.get("OPENAI_API_KEY", "")
        if key:
            os.environ["OPENAI_API_KEY"] = key
    if not os.getenv("OPENAI_MODEL"):
        model = st.secrets.get("OPENAI_MODEL", "")
        if model:
            os.environ["OPENAI_MODEL"] = model
except Exception:  # StreamlitSecretNotFoundError when no secrets file exists
    pass

pages = [st.Page("pages/landing.py", title="서비스 소개", icon="🏠", default=True),
         st.Page("pages/analysis.py", title="모의고사 분석", icon="📊")]
st.navigation(pages).run()
