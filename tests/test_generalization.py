"""New PDF/Excel layouts, independent score oracle, shared passages and isolation."""
import io
import random
import pytest
import pymupdf
from openpyxl import load_workbook
from core.pipeline import run_analysis
from core.pdf_reader import parse_answer_pdf
from core.questions import parse_question_pdf
from core.grading import grade_and_analyze
from core.omr_reader import parse_omr_excel
from core.excel_export import export_results
from core.session_context import init_session, sync_uploads
from tests.factories import key_pdf, omr_xlsx, problem_pdf


@pytest.mark.parametrize('size', [1, 10, 120, 257])
def test_binary_pipeline_against_independent_oracle(size):
    rng = random.Random(791 + size)
    key = {n: {'answer': rng.randint(1, 5), 'points': p} for n, p in enumerate([2, 4, 7, 1], 1)}
    students = [{'name': f'ID-{i:04}', 'answers': {n: rng.choice([None, 1, 2, 3, 4, 5]) for n in key}} for i in range(size)]
    pdf = key_pdf([{'number': n, **k} for n, k in key.items()], order=('points', 'number', 'answer'))
    report = run_analysis(problem_pdf(key), pdf, omr_xlsx(students))
    expected = [sum(key[n]['points'] for n, a in s['answers'].items() if a == key[n]['answer']) for s in students]
    assert [s['score'] for s in report['students']] == expected
    assert report['summary']['mean'] == pytest.approx(sum(expected)/size)
    assert report['summary']['std_population'] ** 2 == pytest.approx(sum((s-sum(expected)/size)**2 for s in expected)/size)
    assert report['summary']['lowest_students'] == [students[i]['name'] for i, s in enumerate(expected) if s == min(expected)]
    book = load_workbook(io.BytesIO(export_results(report)), read_only=True)
    matrix = list(book['학생별 채점 결과'].values)
    assert len(matrix) == size+1
    for row, original, score in zip(matrix[1:], students, expected):
        assert row[1] == score
        assert row[2:] == tuple('O' if original['answers'][n] == key[n]['answer'] else 'X' for n in key)


def test_actual_uploaded_answer_change_changes_scores():
    students = [{'name': 'new-ID', 'answers': {1: 1, 2: 3}}]
    omr, problems = omr_xlsx(students), problem_pdf([1, 2])
    first = run_analysis(problems, key_pdf([{'number': 1, 'answer': 1, 'points': 2}, {'number': 2, 'answer': 3, 'points': 5}]), omr)
    second = run_analysis(problems, key_pdf([{'number': 1, 'answer': 2, 'points': 4}, {'number': 2, 'answer': 3, 'points': 1}]), omr)
    assert first['students'][0]['score'] == 7 and second['students'][0]['score'] == 1


@pytest.mark.parametrize('borders', [False, True])
def test_new_answer_table_layouts(borders):
    raw = key_pdf([{'number': 1, 'answer': 4, 'points': 1.5}, {'number': 2, 'answer': 2, 'points': 8}], borders=borders)
    assert parse_answer_pdf(raw) == {1: {'answer': 4, 'points': 1.5}, 2: {'answer': 2, 'points': 8}}


@pytest.mark.parametrize('rows', [
    [{'number': 1, 'answer': 1, 'points': 2}, {'number': 1, 'answer': 2, 'points': 2}],
    [{'number': 1, 'answer': 1, 'points': 2}, {'number': 3, 'answer': 2, 'points': 2}],
    [{'number': 1, 'answer': 0, 'points': 2}], [{'number': 1, 'answer': 1, 'points': 0}],
])
def test_rejects_bad_answer_pdf(rows):
    with pytest.raises(ValueError):
        parse_answer_pdf(key_pdf(rows))


def test_shared_passage_across_pages_and_incomplete_detection():
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((40, 60), '[1-2] Read the following passage.')
        page.insert_text((40, 90), 'Shared passage begins here with independent words about nature.')
        page = pdf.new_page()
        page.insert_text((40, 50), 'Continued common passage across page boundaries.')
        page.insert_text((40, 90), '1. Choose the title.')
        page.insert_text((40, 120), 'A) a B) b C) c D) d E) e')
        page.insert_text((40, 180), '2. Choose the meaning.')
        page.insert_text((40, 210), 'A) a B) b')
        data = pdf.tobytes()
    questions, missing = parse_question_pdf(data, [1, 2, 3])
    assert missing == [3]
    assert questions[1]['shared_text'] == questions[2]['shared_text']
    assert questions[1]['text'].count('Shared passage begins') == 1
    assert 'Continued common passage' in questions[2]['text']
    assert questions[1]['extraction_status'] == 'complete'
    assert questions[2]['extraction_status'] == 'partial'
    assert {r['page'] for r in questions[1]['regions']} == {1, 2}
    assert '2. Choose' not in questions[1]['text']


def test_vector_chart_region_is_attached_to_its_question():
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((40, 50), '1. What does the chart show about these independent results?')
        page.draw_rect(pymupdf.Rect(60, 80, 180, 160))
        page.insert_text((40, 190), 'A) a B) b C) c D) d E) e')
        page.insert_text((40, 250), '2. The next question is unrelated to the graph above.')
        page.insert_text((40, 280), 'A) a B) b C) c D) d E) e')
        data = pdf.tobytes()
    questions, _ = parse_question_pdf(data, [1, 2])
    assert questions[1]['has_visual']
    assert questions[1]['regions'][-1]['bbox'][3] < 250
    assert questions[1]['regions'][-1]['bbox'][1] < 80


def test_omr_errors_and_formula_export():
    with pytest.raises(ValueError):
        parse_omr_excel(omr_xlsx([{'name': 'duplicate', 'answers': {1: 1}}, {'name': 'duplicate', 'answers': {1: 2}}]), [1])
    with pytest.raises(ValueError):
        parse_omr_excel(omr_xlsx([{'name': 'A', 'answers': {1: 1}}]), [1, 2])
    with pytest.raises(ValueError):
        parse_omr_excel(b'not-an-xlsx', [1])
    report = grade_and_analyze({1: {'answer': 1, 'points': 2}}, [{'name': '=PRIVATE()', 'answers': {1: None}}])
    cell = load_workbook(io.BytesIO(export_results(report)))['학생별 채점 결과']['A2']
    assert cell.data_type == 's' and cell.value.startswith("'=")


def test_upload_deletion_and_session_separation():
    class Upload:
        name = 'input.pdf'
        def __init__(self, data): self.data = data
        def getvalue(self): return self.data
    a, b = {}, {}
    init_session(a); init_session(b)
    sync_uploads(a, [Upload(b'1'), Upload(b'2'), Upload(b'3')])
    a['report'], a['messages'], a['chat_context'] = {'private': 1}, [{'content': 'old'}], {'last_students': ['old']}
    assert not b['messages'] and b['report'] is None
    assert not sync_uploads(a, [Upload(b'1'), Upload(b'2'), Upload(b'3')])
    assert a['report']
    assert sync_uploads(a, [Upload(b'1'), None, Upload(b'3')])
    assert a['report'] is None and not a['messages'] and not a['chat_context'] and a['excel_bytes'] is None
