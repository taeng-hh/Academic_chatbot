"""Local Python timing is explicitly separate from API/browser/cloud E2E."""
import json
import math
from pathlib import Path
from statistics import mean
import time
from core.agent_graph import run_chat
from core.grading import grade_and_analyze
from tests.test_agent import HOLDOUT, report as report_fixture


def test_holdout_local_latency_three_runs():
    # Calling a fixture's underlying factory avoids changing the shared input.
    report = report_fixture.__wrapped__()
    timings = []
    for index, (query, expected) in enumerate(HOLDOUT):
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            answer, _ = run_chat(query, report, [])
            samples.append(time.perf_counter() - start)
            assert all(p in answer for p in expected)
        timings.append({'case': index+1, 'seconds': samples, 'mean': mean(samples),
                        'p95': sorted(samples)[math.ceil(len(samples)*.95)-1]})
    all_samples = [s for row in timings for s in row['seconds']]
    output = {'scope': 'local Python run_chat only; no API, browser rendering or cloud',
              'cases': len(HOLDOUT), 'repetitions': 3, 'passed': len(all_samples),
              'mean_seconds': mean(all_samples), 'p95_seconds': sorted(all_samples)[math.ceil(len(all_samples)*.95)-1],
              'measurements': timings}
    path = Path('output/validation/local_latency.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding='utf-8')


def test_grading_120_separate_latency():
    key = {n: {'answer': (n % 5)+1, 'points': 2} for n in range(1, 46)}
    students = [{'name': f'BENCH-{i}', 'answers': {n: ((i+n) % 5)+1 for n in key}} for i in range(120)]
    durations = []
    for _ in range(3):
        start = time.perf_counter()
        report = grade_and_analyze(key, students)
        durations.append(time.perf_counter() - start)
        assert len(report['students']) == 120
    path = Path('output/validation/grading_latency.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'scope': 'Python grading only, excludes parsing/API/UI', 'seconds': durations,
                                'mean_seconds': mean(durations)}, indent=2), encoding='utf-8')
