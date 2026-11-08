"""Read-only queries over the current session's grading report. No eval or I/O."""
from __future__ import annotations

from collections import Counter
from statistics import mean, median, pstdev


def _stats(students: list[dict]) -> dict:
    scores = [s['score'] for s in students]
    return {'count': len(scores), 'mean': mean(scores) if scores else None,
            'median': median(scores) if scores else None,
            'std_population': pstdev(scores) if scores else None,
            'variance_population': pstdev(scores) ** 2 if scores else None,
            'min': min(scores) if scores else None, 'max': max(scores) if scores else None,
            'score_distribution': dict(sorted(Counter(scores).items()))}


def query_exam(report: dict, operation: str, names: list[str] | None = None,
               numbers: list[int] | None = None, min_score: float | None = None,
               max_score: float | None = None, status: str = 'all', keyword: str = '',
               min_inclusive: bool = True, max_inclusive: bool = True, choice: int | None = None,
               rank: int | None = None, rank_end: int | None = None,
               rank_mode: str = 'competition', rank_from: str = 'top',
               order_by: str = 'score', descending: bool = True,
               limit: int | None = None, include_ties: bool = True) -> dict:
    students = report['students']
    key = report['answer_key']
    numbers = sorted(key) if numbers is None else numbers
    unknown = sorted(set(numbers) - set(key))
    if unknown:
        return {'error': '존재하지 않는 문항 번호', 'numbers': unknown}
    if names is not None:
        missing = sorted(set(names) - {s['name'] for s in students})
        if missing:
            return {'error': '존재하지 않는 학생명. students로 이름을 먼저 확인하세요.', 'names': missing}
    selected = [s for s in students if (names is None or s['name'] in names)
                and (min_score is None or (s['score'] >= min_score if min_inclusive else s['score'] > min_score))
                and (max_score is None or (s['score'] <= max_score if max_inclusive else s['score'] < max_score))]
    if status not in {'all', 'correct', 'wrong', 'blank'}:
        return {'error': '알 수 없는 정오답 조건'}
    if status != 'all':
        selected = [s for s in selected if all(
            s['answers'][n] is None if status == 'blank' else
            bool(s['correct'][n]) == (status == 'correct') for n in numbers)]
    if choice is not None:
        if type(choice) is not int or not 1 <= choice <= 5:
            return {'error': '보기 번호가 잘못되었습니다.'}
        selected = [s for s in selected if all(s['answers'][n] == choice for n in numbers)]

    ranks, offset = {}, 1
    score_counts = Counter(s['score'] for s in students)
    for score, count in sorted(score_counts.items(), reverse=True):
        ranks[score] = offset
        offset += count
    dense_ranks = {score: index for index, score in enumerate(sorted(score_counts, reverse=True), 1)}
    bottom_ranks, offset = {}, 1
    for score, count in sorted(score_counts.items()):
        bottom_ranks[score] = offset
        offset += count
    bottom_dense = {score: index for index, score in enumerate(sorted(score_counts), 1)}
    if rank_mode not in {'competition', 'dense'} or rank_from not in {'top', 'bottom'}:
        return {'error': '지원하지 않는 순위 기준입니다.'}
    if rank is not None:
        if type(rank) is not int or rank < 1 or (rank_end is not None and (type(rank_end) is not int or rank_end < rank)):
            return {'error': '순위 범위가 잘못되었습니다.'}
        rank_values = (dense_ranks if rank_mode == 'dense' else ranks) if rank_from == 'top' else (bottom_dense if rank_mode == 'dense' else bottom_ranks)
        selected = [s for s in selected if rank <= rank_values[s['score']] <= (rank_end or rank)]

    def student_row(s, details=False):
        row = {'name': s['name'], 'score': s['score'],
               'rank': ranks[s['score']],
               'score_rank': dense_ranks[s['score']],
               'wrong_numbers': [n for n in key if not s['correct'][n]],
               'blank_numbers': [n for n in key if s['answers'][n] is None]}
        row['correct_count'] = len(key) - len(row['wrong_numbers'])
        row['lost_points'] = sum(key[n]['points'] for n in row['wrong_numbers'])
        if details:
            row['responses'] = [{'number': n, 'submitted_answer': s['answers'][n],
                                 **key[n], 'correct': s['correct'][n]} for n in numbers]
        return row

    if operation == 'overview':
        return {'summary': report['summary'], 'statistics': _stats(selected),
                'missing_questions': report.get('missing_questions', [])}
    if operation in {'students', 'student'}:
        if order_by not in {'score', 'name'}:
            return {'error': '학생 정렬 기준은 점수 또는 이름이어야 합니다.'}
        ordered = sorted(selected, key=lambda s: s['name'])
        ordered.sort(key=lambda s: s[order_by], reverse=descending)
        shown = ordered
        if limit is not None:
            if type(limit) is not int or limit < 1:
                return {'error': '조회 개수가 잘못되었습니다.'}
            shown = ordered[:limit]
            if include_ties and order_by == 'score' and shown:
                shown += [s for s in ordered[limit:] if s['score'] == shown[-1]['score']]
        return {'statistics': _stats(selected), 'student_count_total': len(students),
                'students': [student_row(s, operation == 'student') for s in shown],
                'shown_count': len(shown), 'rank_mode': rank_mode, 'rank_from': rank_from}
    if operation == 'questions':
        rows = []
        for n in numbers:
            misses = sum(not s['correct'][n] for s in selected)
            rows.append({'number': n, **key[n], 'student_count': len(selected),
                         'miss_count': misses, 'correct_count': len(selected) - misses,
                         'blank_count': sum(s['answers'][n] is None for s in selected),
                         'miss_rate': misses / len(selected) if selected else None})
        return {'questions': rows}
    if operation == 'question':
        results = []
        for n in numbers:
            source = report.get('questions', {}).get(n)
            results.append({'number': n, 'source': source,
                            'source_status': 'extracted' if source else 'PDF 문항 원문 추출 실패',
                            **key[n],
                            'choice_counts': dict(Counter(str(s['answers'][n]) if s['answers'][n] is not None else 'blank' for s in selected)),
                            'responses': [{'name': s['name'], 'answer': s['answers'][n],
                                           'correct': s['correct'][n]} for s in selected]})
        return {'questions': results}
    if operation == 'search':
        if not keyword.strip():
            return {'error': '검색 키워드가 필요합니다'}
        return {'questions': [q for n, q in report.get('questions', {}).items()
                              if n in numbers and keyword.casefold() in q['text'].casefold()]}
    return {'error': '알 수 없는 조회 작업'}
