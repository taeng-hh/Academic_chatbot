"""Pure Python preprocessing pipeline, run once on Analysis button."""
from __future__ import annotations
from .pdf_reader import parse_answer_pdf
from .omr_reader import parse_omr_excel
from .grading import grade_and_analyze
from .questions import parse_question_pdf


def run_analysis(problem_pdf: bytes, answer_pdf: bytes, omr_xlsx: bytes) -> dict:
    key = parse_answer_pdf(answer_pdf)
    students = parse_omr_excel(omr_xlsx, sorted(key))
    report = grade_and_analyze(key, students)
    questions, missing = parse_question_pdf(problem_pdf, sorted(key))
    report["questions"] = questions
    report["missing_questions"] = missing
    report['incomplete_questions'] = [n for n, q in questions.items() if q.get('extraction_status') == 'partial']
    return report
