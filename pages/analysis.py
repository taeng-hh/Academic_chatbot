"""Grading page with upload sidebar, Excel download, and session-local chat."""
from __future__ import annotations

import os
import time
import streamlit as st

from core.pipeline import run_analysis
from core.excel_export import export_results
from core.agent_graph import run_chat
from core.openai_service import generate_similar_question, DEFAULT_MODEL
from core.session_context import init_session, reset_analysis, sync_uploads


init_session(st.session_state)
st.title("📊 모의고사 분석 및 AI 채팅")
api_key = os.getenv("OPENAI_API_KEY", "")
model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

with st.sidebar:
    st.header("입력 파일")
    problem_file = st.file_uploader("① 영어 문제 PDF", type=["pdf"], key="problem_pdf")
    answer_file = st.file_uploader("② 정답 및 배점 PDF", type=["pdf"], key="answer_pdf")
    omr_file = st.file_uploader("③ OMR 결과 Excel", type=["xlsx"], key="omr_xlsx")
    ready = all((problem_file, answer_file, omr_file))
    sync_uploads(st.session_state, (problem_file, answer_file, omr_file))
    pre_generate = st.checkbox("오답 1위 유사 문제 미리 생성", value=False,
                                help="분석할 때 OpenAI API를 한 번 호출합니다. 채팅에서는 미리 생성된 문제를 바로 표시합니다.")
    analyze_pressed = st.button("분석 시작", type="primary", use_container_width=True,
                               disabled=not ready)
    if analyze_pressed and ready:
        try:
            with st.spinner("PDF·Excel 분석 및 자동 채점 중..."):
                report = run_analysis(problem_file.getvalue(), answer_file.getvalue(),
                                      omr_file.getvalue())
            st.session_state.report = report
            st.session_state.question_pdf_bytes = problem_file.getvalue()
            st.session_state.excel_bytes = None
            st.session_state.cached_question = None
            st.session_state.messages = []
            st.session_state.last_student = None
            st.session_state.chat_context = {}
            if pre_generate and api_key:
                best_no = report["summary"]["most_missed"][0]["number"]
                try:
                    with st.spinner("오답률 1위 유사 문제 미리 생성 중..."):
                        st.session_state.cached_question = generate_similar_question(
                            report["questions"][best_no], api_key, model, st.session_state.question_pdf_bytes)
                except Exception as exc:
                    st.warning(f"채점은 완료됐으나 유사 문제 미리 생성은 실패했습니다: {type(exc).__name__}")
            st.success("자동 채점 및 분석이 완료되었습니다.")
        except Exception as exc:
            reset_analysis(st.session_state)
            st.error(f"분석에 실패했습니다: {exc}" if isinstance(exc, ValueError) else "파일 처리에 실패했습니다. PDF/Excel 형식과 파일 손상 여부를 확인해 주세요.")
    if st.session_state.report is not None:
        try:
            # Build from the current report and exporter on every rerun; a
            # previously serialized workbook may predate a code update.
            st.session_state.excel_bytes = export_results(st.session_state.report)
            st.download_button("📥 Excel 추출", data=st.session_state.excel_bytes,
                               file_name="모의고사_채점결과.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
            st.caption('Excel은 “학생별 채점 결과” 시트가 먼저 열립니다. 같은 파일 하단의 “전체 성적 개요” 시트에는 중앙값·평균·표준편차·오답 상위 3문항이 함께 저장됩니다.')
        except Exception:
            st.session_state.excel_bytes = None
            st.error('Excel 생성에 실패했습니다. 현재 채점 결과와 파일 형식을 확인해 주세요.')
    else:
        st.button("📥 Excel 추출", disabled=True, use_container_width=True)
    if api_key:
        st.caption(f"OpenAI 사용 가능 · 모델: {model}")
    else:
        st.caption("OPENAI_API_KEY 미설정: 자동 채점·Excel은 가능하나 LLM 응답은 제한됩니다.")

report = st.session_state.report
if report is None:
    st.info("왼쪽에서 문제 PDF, 정답 PDF, OMR Excel을 업로드하고 **분석 시작**을 클릭하세요.")
else:
    if report["missing_questions"]:
        st.warning(f"문제 PDF에서 다음 문항을 추출하지 못했습니다: {report['missing_questions']}. 성적 채점은 정상 완료됐습니다.")
    if report.get('incomplete_questions'):
        st.warning(f"원문·선택지 추출 확인이 필요한 문항: {report['incomplete_questions']}. 해당 문항의 문제 생성은 제한됩니다.")
    with st.expander("학생별 채점 결과 미리 보기 (총점·문항별 O/X)"):
        import pandas as pd
        numbers = sorted(report['answer_key'])
        st.dataframe(pd.DataFrame([
            {'학생명': r['name'], '총점': r['score'],
             **{f'Q{n:02d}': 'O' if r['correct'][n] else 'X' for n in numbers}}
            for r in report['students']]),
                     use_container_width=True, hide_index=True)
        st.caption('O = 정답 / X = 오답(미응답 포함). 문항이 많으면 표를 오른쪽으로 스크롤하세요.')
    st.divider()
    st.subheader("💬 AI 평가 도우미")
    st.caption("예: 전체 성적 요약 / 2등 학생과 점수 / 상위 5명의 평균 / 점수 구간별 인원 / 학생 간 오답 비교 / 문항 해설 / 그 학생의 오답으로 유사 문제 만들기")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

prompt = st.chat_input("성적에 관해 질문해 보세요", disabled=report is None,
                       max_chars=1200)
if prompt and report is not None:
    started = time.perf_counter()
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("답변 작성 중..."):
            try:
                reply, selected_student = run_chat(
                    prompt, report, st.session_state.messages, api_key, model,
                    st.session_state.cached_question, st.session_state.question_pdf_bytes,
                    st.session_state.last_student, context=st.session_state.chat_context)
            except Exception as exc:
                reply, selected_student = f"채팅 처리 오류: {type(exc).__name__}. 로그를 확인해 주세요.", None
        st.markdown(reply)
        st.session_state.last_latency = time.perf_counter() - started
        st.caption(f"응답 처리: {st.session_state.last_latency:.2f}초")
    st.session_state.messages.extend([
        {"role": "user", "content": prompt}, {"role": "assistant", "content": reply}])
    st.session_state.messages = st.session_state.messages[-30:]
    st.session_state.last_student = selected_student
