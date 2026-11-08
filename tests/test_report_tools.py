"""Data queries are deterministic and do not need network access."""
import pytest

from core.grading import grade_and_analyze
from core.report_tools import query_exam


@pytest.fixture
def report():
    report = grade_and_analyze(
        {1: {'answer': 1, 'points': 2}, 2: {'answer': 3, 'points': 3}},
        [{'name': 'A', 'answers': {1: 1, 2: 3}},
         {'name': 'B', 'answers': {1: 1, 2: 3}},
         {'name': 'C', 'answers': {1: 2, 2: None}}])
    report['questions'] = {1: {'number': 1, 'page': 1, 'text': 'Digital platforms connect people.'}}
    return report


def test_ranks_filters_and_aggregate(report):
    rows = query_exam(report, 'students')['students']
    assert [s['rank'] for s in rows] == [1, 1, 3]
    filtered = query_exam(report, 'students', min_score=5, max_score=5)
    assert filtered['statistics']['count'] == 2
    assert filtered['statistics']['mean'] == 5
    assert query_exam(report, 'students', min_score=6)['statistics']['mean'] is None
    assert query_exam(report, 'overview')['statistics']['variance_population'] == pytest.approx(50 / 9)


def test_individual_answers_and_question_distribution(report):
    student = query_exam(report, 'student', names=['C'])['students'][0]
    assert student['wrong_numbers'] == [1, 2]
    assert student['blank_numbers'] == [2]
    assert student['lost_points'] == 5
    assert student['responses'][1]['submitted_answer'] is None
    item = query_exam(report, 'question', numbers=[2])['questions'][0]
    assert item['source'] is None
    assert item['choice_counts'] == {'3': 2, 'blank': 1}
    assert query_exam(report, 'students', numbers=[2], status='blank')['students'][0]['name'] == 'C'
    assert query_exam(report, 'students', numbers=[1, 2], status='correct')['statistics']['count'] == 2
    assert query_exam(report, 'questions', names=['C'], numbers=[1])['questions'][0]['miss_rate'] == 1


def test_search_and_unknown_references(report):
    assert query_exam(report, 'search', keyword='PLATFORMS')['questions'][0]['number'] == 1
    assert query_exam(report, 'question', numbers=[999])['error']
    assert query_exam(report, 'student', names=['unknown'])['error']
    assert query_exam(report, '__import__')['error']
