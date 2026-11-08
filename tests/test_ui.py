"""Streamlit AppTest: both pages, controls, repeated chat and independent sessions."""
from streamlit.testing.v1 import AppTest
from pathlib import Path
from core.excel_export import export_results
from tests.test_agent import report as report_fixture
ROOT = Path(__file__).resolve().parents[1]


def test_landing_page_starts():
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=15)
    assert not app.exception
    assert any('평가 도우미' in title.value for title in app.title)


def test_analysis_chat_and_session_isolation(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    a = AppTest.from_file(str(ROOT / 'pages/analysis.py')).run(timeout=15)
    b = AppTest.from_file(str(ROOT / 'pages/analysis.py')).run(timeout=15)
    assert not a.exception and not b.exception
    assert a.chat_input[0].disabled
    assert not a.checkbox[0].value  # pre-generation remains optional and off
    report = report_fixture.__wrapped__()
    report['missing_questions'], report['incomplete_questions'] = [], []
    a.session_state['report'] = report
    a.session_state['excel_bytes'] = export_results(report)
    a.run(timeout=15)
    assert not a.exception and not a.chat_input[0].disabled
    a.chat_input[0].set_value('하잉').run(timeout=15)
    assert not a.exception
    assert '안녕하세요' in a.session_state['messages'][-1]['content']
    a.chat_input[0].set_value('김서윤 점수는?').run(timeout=15)
    assert not a.exception and '10점' in a.session_state['messages'][-1]['content']
    assert b.session_state['report'] is None and not b.session_state['messages']
