"""Tests run offline with the files supplied in this conversation.
For tests on your PC set TEST_DATA_DIR to the folder with the 3 input files.
"""
from __future__ import annotations
import os
from pathlib import Path
import io
import pytest
import pandas as pd

from core.pdf_reader import choice_number, parse_answer_pdf
from core.omr_reader import parse_omr_excel
from core.grading import grade_and_analyze
from core.excel_export import export_results
from core.questions import parse_question_pdf

DATA = Path(os.environ.get("TEST_DATA_DIR", "tests/data"))
ANS = DATA / "2026학년도-대학수학능력시험-영어-정답.pdf"
PROBLEM = DATA / "2026학년도-대학수학능력시험-영어-문제.pdf"
OMR = DATA / "sample_data.xlsx"


def test_choice_normalization():
    assert choice_number("③") == 3
    assert choice_number(2.0) == 2
    assert choice_number("5") == 5
    assert choice_number(None) is None
    with pytest.raises(ValueError):
        choice_number(8)


@pytest.mark.parametrize('two_columns', [False, True])
def test_question_layout_keeps_passage_and_choices(two_columns):
    import pymupdf
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=600, height=800)
        page.insert_text((40, 50), '21. Choose the correct answer.')
        passage = ('Digital platforms connect people across many countries and communities.'
                   if not two_columns else 'Digital platforms connect people.')
        page.insert_text((40, 75), passage, fontsize=10)
        page.insert_text((40, 95), 'A) First option B) Second option', fontsize=10)
        page.insert_text((330 if two_columns else 40, 50 if two_columns else 150),
                         '22. Next question.')
        content = pdf.tobytes()
    questions, missing = parse_question_pdf(content, [21, 22])
    assert not missing
    assert passage in questions[21]['text']
    assert 'Second option' in questions[21]['text']
    assert '22. Next' not in questions[21]['text']


def test_grade_synthetic():
    key = {1: {"answer": 2, "points": 2}, 2: {"answer": 1, "points": 3}}
    students = [{"name": "A", "answers": {1: 2, 2: 1}},
                {"name": "B", "answers": {1: 1, 2: None}}]
    report = grade_and_analyze(key, students)
    assert [r["score"] for r in report["students"]] == [5, 0]
    assert report["summary"]["mean"] == 2.5
    assert report["summary"]["median"] == 2.5
    assert report["summary"]["std_population"] == 2.5
    assert report["summary"]["lowest_students"] == ["B"]
    assert report["summary"]["most_missed"][0]["number"] == 1
    binary = export_results(report)
    assert len(binary) > 1000
    from openpyxl import load_workbook
    xls = load_workbook(io.BytesIO(binary), read_only=True)
    assert xls.sheetnames == ["전체 성적 개요", "학생별 채점 결과", "문항별 분석", '학생별 석차 및 답안']


@pytest.mark.skipif(not (ANS.exists() and OMR.exists()), reason="uploaded sample not available")
def test_real_sample_grades():
    key = parse_answer_pdf(ANS.read_bytes())
    students = parse_omr_excel(OMR.read_bytes(), sorted(key))
    report = grade_and_analyze(key, students)
    s = report["summary"]
    assert len(key) == 45 and s["max_score"] == 100
    assert s["student_count"] == 10
    assert s["mean"] == 65.2
    assert s["median"] == 65.5
    assert round(s["std_population"], 2) == 19.58
    assert (s["lowest_students"], s["lowest_score"]) == (["909학생"], 31)
    assert [x["number"] for x in s["most_missed"]] == [21, 33, 42]


@pytest.mark.skipif(not (ANS.exists() and PROBLEM.exists()), reason="uploaded sample not available")
def test_question_extraction():
    key = parse_answer_pdf(ANS.read_bytes())
    questions, missing = parse_question_pdf(PROBLEM.read_bytes(), sorted(key))
    assert not missing
    assert len(questions) == 45
    assert "Digital platforms" in questions[21]["text"]
    assert "Giving clients" in questions[33]["text"]
    assert "dress" in questions[42]["text"]
    assert all(questions[n]['shared_text'].strip() for n in range(41, 46))
    assert questions[41]['shared_text'] == questions[42]['shared_text']
    assert questions[43]['shared_text'] == questions[45]['shared_text']
    assert all(questions[n]['choice_count'] == 5 for n in [21, 33, 42])
    assert all(questions[n]['extraction_status'] != 'partial' for n in [21, 33, 42])
