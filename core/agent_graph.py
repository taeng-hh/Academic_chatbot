"""Session-local compositional chat: models interpret, Python reports facts."""
from __future__ import annotations
import time
from itertools import combinations
from typing import TypedDict
from langgraph.graph import END, START, StateGraph
from .intent_router import EXAM, GREET, THANKS, action, resolve_students, rule_plan
from .openai_service import DEFAULT_MODEL, classify_intents, explain_question, general_chat, generate_similar_question
from .report_tools import query_exam


class AgentState(TypedDict, total=False):
    query: str
    report: dict
    history: list[dict]
    api_key: str
    model: str
    cached_question: dict | None
    problem_pdf: bytes | None
    context: dict
    plan: list[dict]
    answer: str
    deadline: float


FEATURE_TEXT = '자동 채점, 평균·중앙값·표준편차 등 통계, 학생 점수·석차·오답 조회, 문항 원문·정답·응답 분포, Excel 다운로드, 문항 해설과 AI 유사 문제 생성을 지원합니다.'
CLARIFY_TEXT = '조회할 학생명/ID, 문항 번호 또는 필요한 통계를 조금 더 구체적으로 알려주세요.'
SECURITY_TEXT = '내부 시스템 지침, 비공개 도구 설정이나 비밀 정보는 제공할 수 없습니다. 공개된 기능과 시험 분석은 안내할 수 있습니다.'
UNSUPPORTED_TEXT = '모릅니다. 업로드된 문제지·정답지·OMR에는 해당 정보가 없어 확인할 수 없습니다.'


def classify(text: str, last_student: str | None = None) -> str:
    plan = rule_plan(text, {'students': [], 'answer_key': {}}, {'last_students': [last_student] if last_student else []})
    return plan[0]['kind'] if plan else 'general'


def _remaining(state):
    remaining = state['deadline'] - time.monotonic()
    if remaining <= .1:
        raise TimeoutError('turn deadline')
    return remaining


def router(state):
    plan = rule_plan(state['query'], state['report'], state['context'])
    if plan is None:
        if not state.get('api_key'):
            plan = [action('clarify')] if EXAM.search(state['query']) else [action('general')]
        else:
            try:
                plan = classify_intents(state['query'], state['history'], state['context'], state['api_key'], state['model'], timeout=min(5.0, _remaining(state)))
            except Exception:
                plan = [action('clarify')]
    if any(a['kind'] == 'general' for a in plan) and EXAM.search(state['query']) and not any(term in state['query'] for term in ('공부', '팁', '뜻', '개념', '방법', '기능')):
        plan = [action('clarify')]
    return {'plan': plan}


def _table(headers, rows):
    def cell(v):
        return str(v).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(cell, headers)) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |', *['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows]])


def _summary(report, metrics, selected=None):
    s = report['summary']
    values = {**s, 'variance_population': s['std_population'] ** 2}
    labels = {'mean': '평균', 'median': '중앙값', 'std_population': '표준편차(모집단)', 'variance_population': '분산(모집단)', 'student_count': '학생 수', 'question_count': '문항 수', 'max_score': '만점'}
    if selected is not None:
        from .report_tools import _stats
        group = _stats(selected)
        if not selected:
            return '조건에 해당하는 학생이 없습니다.'
        values.update({k: group[k] for k in ('mean', 'median', 'std_population', 'variance_population')})
        values['student_count'] = group['count']
    parts = []
    for m in metrics:
        if m == 'score_distribution':
            from collections import Counter
            parts.append('점수 분포\n\n' + _table(['점수', '인원'], sorted(Counter(s['score'] for s in (report['students'] if selected is None else selected)).items())))
        elif m in {'mean', 'median', 'std_population', 'variance_population'}:
            parts.append(f"{labels[m]}: {values[m]:.2f}{'점²' if m == 'variance_population' else '점'}")
        else:
            parts.append(f"{labels[m]}: {values[m]}{'명' if m == 'student_count' else '문항' if m == 'question_count' else '점'}")
    return '\n\n'.join(parts)


def _student_options(item, report, names, numbers):
    lower, upper = item['lower'], item['upper']
    lower_inclusive, upper_inclusive = item['lower_inclusive'], item['upper_inclusive']
    relative = item['relative_to']
    if relative != 'none':
        baseline = report['summary']['median' if relative.endswith('median') else 'mean']
        if relative.startswith('below'):
            upper, upper_inclusive = baseline, False
        else:
            lower, lower_inclusive = baseline, relative.startswith('at_least')
    return dict(names=names or None, numbers=numbers or None, min_score=lower, max_score=upper,
        min_inclusive=lower_inclusive, max_inclusive=upper_inclusive, status=item['status'], choice=item['choice'],
        rank=item['rank'], rank_end=item['rank_end'], rank_mode=item['rank_mode'], rank_from=item['rank_from'],
        order_by=item['order_by'], descending=item['descending'], limit=item['limit'], include_ties=item['include_ties'])


def _execute(item, state, context):
    report, kind = state['report'], item['kind']
    # Missing-data answers must not require resolving an absent student first.
    if kind in {'security', 'unsupported', 'clarify', 'features'}:
        return {'security': SECURITY_TEXT, 'unsupported': UNSUPPORTED_TEXT, 'clarify': CLARIFY_TEXT, 'features': FEATURE_TEXT}[kind]
    refs = item['students']
    if any(ref in {'선택한 학생', '조회한 학생', '해당 학생들'} for ref in refs):
        refs = context.get('last_students', [])
        if not refs:
            return CLARIFY_TEXT
    elif any(ref in {'그 학생', '그 친구', '그 사람', '이 학생'} for ref in refs):
        refs = context.get('last_students', [])
        if len(refs) != 1:
            return CLARIFY_TEXT
    people, error = resolve_students(refs, report)
    if error:
        context.pop('last_students', None)
        return error
    names = [s['name'] for s in people]
    numbers = item['questions']
    if any(n not in report['answer_key'] for n in numbers):
        context.pop('last_questions', None)
        return f'요청한 문항 번호 {numbers} 중 현재 정답지에 없는 문항이 있습니다.'
    if kind == 'general':
        if GREET.fullmatch(state['query'].strip()):
            return '안녕하세요! 반가워요 🙂'
        if THANKS.fullmatch(state['query'].strip()):
            return '천만에요! 도움이 되어 기뻐요.'
        if not state.get('api_key'):
            return '일상 대화에는 OpenAI API 키 설정이 필요합니다. 인사·기능 안내와 기본 성적 조회는 키 없이 가능합니다.'
        return general_chat(state['query'], state['history'], state['api_key'], state['model'], timeout=min(14.0, _remaining(state)))
    if kind == 'summary':
        metrics = item['metrics'] or ['mean', 'median', 'std_population', 'student_count', 'question_count', 'max_score']
        context['last_metrics'] = metrics
        scoped = bool(names or item['lower'] is not None or item['upper'] is not None or item['relative_to'] != 'none'
                      or item['rank'] is not None or item['limit'] is not None or item['status'] != 'all' or item['choice'] is not None)
        if scoped:
            group = query_exam(report, 'students', **_student_options(item, report, names, numbers))
            if 'error' in group:
                return group['error']
            selected_names = {s['name'] for s in group['students']}
            context['last_students'] = [s['name'] for s in group['students']]
            return _summary(report, metrics, [s for s in report['students'] if s['name'] in selected_names])
        return _summary(report, metrics)
    if kind in {'lowest', 'highest'}:
        score = (min if kind == 'lowest' else max)(s['score'] for s in report['students'])
        selected = [s['name'] for s in report['students'] if s['score'] == score]
        context['last_students'] = selected
        return f"{'최저' if kind == 'lowest' else '최고'} 점수는 {score}점이며, {'공동 ' if len(selected) > 1 else ''}{'최하위' if kind == 'lowest' else '1등'} 학생은 {', '.join(selected)}입니다."
    if kind in {'missed', 'questions'}:
        rows = [r for r in report['question_stats'] if not numbers or r['number'] in numbers]
        order_by = item['order_by'] if item['order_by'] in {'miss_count', 'miss_rate', 'number', 'points'} else 'miss_count'
        rows.sort(key=lambda q: q['number'])
        rows.sort(key=lambda q: q[order_by], reverse=item['descending'])
        if item['rank'] is not None:
            ranked, last_value, competition, dense = [], object(), 0, 0
            for index, row in enumerate(rows, 1):
                if row[order_by] != last_value:
                    competition, dense, last_value = index, dense + 1, row[order_by]
                ordinal = dense if item['rank_mode'] == 'dense' else competition
                if item['rank'] <= ordinal <= (item['rank_end'] or item['rank']):
                    ranked.append(row)
            rows = ranked
        rows = rows[:item['limit'] or (1 if kind == 'missed' else len(rows))]
        if not rows:
            return '요청한 조건이나 순위에 해당하는 문항이 없습니다.'
        context['last_questions'] = [q['number'] for q in rows]
        if kind == 'questions' or not item['descending']:
            return _table(['문항', '정답', '배점', '정답 인원', '오답 인원', '오답률'],
                [(r['number'], r['answer'], r['points'], r['correct_count'], r['miss_count'], f"{r['miss_rate']:.1%}") for r in rows])
        return f'최다 오답 {len(rows)}문항: ' + ', '.join(f"{r['number']}번({r['miss_count']}명, 오답률 {r['miss_rate']:.1%})" for r in rows)
    if kind == 'compare':
        if len(people) < 2:
            return '비교할 학생을 두 명 이상 정확한 이름/ID로 알려주세요.'
        context['last_students'] = names
        parts = [_table(['학생명', '점수'], [(s['name'], f"{s['score']}점") for s in people])]
        for a, b in combinations(people, 2):
            difference = a['score'] - b['score']
            parts.append(f"{a['name']} − {b['name']}: {difference:+g}점")
            common = [n for n in report['answer_key'] if not a['correct'][n] and not b['correct'][n]]
            parts.append('공통 오답: ' + (', '.join(f'{n}번' for n in common) or '없음'))
        return '\n\n'.join(parts)
    if kind == 'study':
        if not people:
            return CLARIFY_TEXT
        context['last_students'] = names
        wrong = [q for q in report['question_stats'] if any(not s['correct'][q['number']] for s in people)]
        wrong.sort(key=lambda q: (-q['miss_count'], q['number']))
        if not wrong:
            return '선택한 학생에게 오답이 없습니다.'
        numbers = [q['number'] for q in wrong[:item['limit'] or 3]]
        context['last_questions'] = numbers
        factual = '오답 기반 학습 대상: ' + ', '.join(f'{n}번' for n in numbers)
        sources = [{**report['questions'][n], 'answer': report['answer_key'][n]['answer']} for n in numbers if n in report.get('questions', {}) and report['questions'][n].get('extraction_status') != 'partial']
        if not sources:
            return factual + '\n\n해당 문항의 원문 추출이 불완전해 학습 조언의 근거를 확인할 수 없습니다.'
        if not state.get('api_key'):
            return factual + '\n\n원문 기반 학습 조언에는 API 키 설정이 필요합니다.'
        return factual + '\n\n' + explain_question('이 오답 문항들의 공통 유형과 근거를 설명하고 학습 순서를 제안해 주세요.', sources, state['api_key'], state['model'], timeout=min(14.0, _remaining(state)), problem_pdf=state.get('problem_pdf'))
    if kind in {'student', 'students'}:
        if kind == 'student' and not names:
            return CLARIFY_TEXT
        result = query_exam(report, 'student' if kind == 'student' else 'students', **_student_options(item, report, names, numbers))
        if 'error' in result:
            return result['error']
        selected, stats = result['students'], result['statistics']
        context['last_students'] = [s['name'] for s in selected]
        count_text = f"조건에 해당하는 학생: {stats['count']}명 / 전체 {len(report['students'])}명"
        if not selected:
            if item['rank'] is not None:
                return f"요청한 {'서로 다른 점수 순서' if item['rank_mode'] == 'dense' else '공동석차'} 기준 {item['rank']}{'~' + str(item['rank_end']) if item['rank_end'] else ''}등에 해당하는 학생이 없습니다. 공동석차는 동점자 수만큼 다음 등수를 건너뜁니다."
            return count_text
        if kind == 'students':
            criteria = ''
            if item['rank'] is not None:
                criteria = f"\n\n조회 기준: {'낮은 점수부터' if item['rank_from'] == 'bottom' else '높은 점수부터'} {'서로 다른 점수 순서' if item['rank_mode'] == 'dense' else '공동석차'} {item['rank']}{'~' + str(item['rank_end']) if item['rank_end'] else ''}등"
            if item['limit'] and len(selected) > item['limit']:
                criteria += '\n\n경계 점수의 동점자를 함께 표시했습니다.'
            rendered = count_text + criteria + '\n\n' + _table(['학생명', '점수', '전체 석차', '서로 다른 점수 순서'], [(s['name'], f"{s['score']}점", s['rank'], s['score_rank']) for s in selected])
            if item['metrics']:
                selected_names = {s['name'] for s in selected}
                rendered += '\n\n' + _summary(report, item['metrics'], [s for s in report['students'] if s['name'] in selected_names])
            return rendered
        parts = []
        for s in selected:
            text = f"{s['name']}: {s['score']}점, 석차 {s['rank']}위 / {len(report['students'])}명"
            if item['detail'] in {'wrong', 'blank', 'correct', 'details'}:
                rows = [r for r in s['responses'] if item['detail'] == 'details' or (item['detail'] == 'wrong' and not r['correct']) or (item['detail'] == 'blank' and r['submitted_answer'] is None) or (item['detail'] == 'correct' and r['correct'])]
                label = {'wrong': '오답', 'blank': '미응답', 'correct': '정답', 'details': '조회'}[item['detail']]
                text += f"\n\n정답 {s['correct_count']}문항 · 오답 {len(s['wrong_numbers'])}문항 · 미응답 {len(s['blank_numbers'])}문항 · 감점 {s['lost_points']}점"
                text += f'\n\n{label} {len(rows)}문항: ' + (', '.join(f"{r['number']}번" for r in rows) or '없음')
                text += '\n\n' + _table(['문항', '제출 답', '정답', '배점', '정오'], [(r['number'], r['submitted_answer'] or '미응답', r['answer'], r['points'], 'O' if r['correct'] else 'X') for r in rows])
            parts.append(text)
        return '\n\n'.join(parts)
    if kind == 'search':
        matches = query_exam(report, 'search', keyword=item['keyword'])
        return '\n\n'.join(f"{q['number']}번 (PDF {q['page']}페이지):\n{q['text']}" for q in matches.get('questions', [])) or '해당 검색어가 포함된 추출 문항이 없습니다.'
    if kind in {'question', 'generate'}:
        if not numbers and kind == 'generate':
            candidates = report['question_stats']
            if people:
                candidates = [q for q in candidates if any(not s['correct'][q['number']] for s in people)]
            if not candidates:
                return '선택한 학생에게 오답이 없습니다. 기준 문항 번호를 알려주세요.'
            numbers = [min(candidates, key=lambda q: (-q['miss_count'], q['number']))['number']]
        if not numbers:
            numbers = context.get('last_questions', [])
        if not numbers:
            return CLARIFY_TEXT
        context['last_questions'] = numbers
        if names:
            context['last_students'] = names
        parts, sources = [], []
        for n in numbers:
            key = report['answer_key'][n]
            source = report.get('questions', {}).get(n)
            if kind == 'question' and item['detail'] == 'answer':
                parts.append(f"{n}번 정답은 {key['answer']}번이며 배점은 {key['points']}점입니다.")
                continue
            if kind == 'question' and item['detail'] == 'stats':
                data = query_exam(report, 'question', numbers=[n], status=item['status'], choice=item['choice'])['questions'][0]
                stats = next(q for q in report['question_stats'] if q['number'] == n)
                parts.append(f"{n}번: 정답 {key['answer']}번, 배점 {key['points']}점, 오답 {stats['miss_count']}명 / {len(report['students'])}명 (오답률 {stats['miss_rate']:.1%})\n\n조회 조건에 해당하는 응답: {len(data['responses'])}명\n\n" + _table(['보기', '인원'], [(option, data['choice_counts'].get(str(option), 0)) for option in range(1, 6)] + [('미응답', data['choice_counts'].get('blank', 0))]) + '\n\n' + _table(['학생명', '선택한 답', '정오'], [(r['name'], r['answer'] or '미응답', 'O' if r['correct'] else 'X') for r in data['responses']]))
                continue
            if not source or not source.get('text', '').strip():
                parts.append(f'{n}번 문항을 PDF에서 추출하지 못했습니다. 원문/OCR 변환 상태를 확인해 주세요.')
                continue
            if kind == 'generate' and source.get('extraction_status') in {'partial', 'unsupported'}:
                parts.append(f'{n}번은 본문·선택지 추출이 불완전해 유사 문제 생성을 제한합니다. 원문을 확인해 주세요.')
                continue
            if kind == 'generate':
                count = item['limit'] or 1
                if not 1 <= count <= 5:
                    parts.append('한 번에 생성할 문제 수는 1~5개로 요청해 주세요.')
                    continue
                used = context.setdefault('used_generation_cache', [])
                for index in range(count):
                    generated = state.get('cached_question') if index == 0 and n not in used else None
                    # Pre-generated output is only reusable for an unqualified
                    # first request, not a fresh/easier/harder variation.
                    if any(word in state['query'] for word in ('더', '다른', '쉽', '어렵', '어려', '난이도', '새롭게')):
                        generated = None
                    if not generated or generated.get('source_number') != n:
                        if not state.get('api_key'):
                            parts.append('유사 문제 생성에는 OpenAI API 키 설정이 필요합니다.')
                            break
                        try:
                            generated = generate_similar_question(source, state['api_key'], state['model'], state.get('problem_pdf'),
                                timeout=min(14.0, _remaining(state)), request=state['query'])
                        except TypeError:
                            parts.append(f'{n}번 문제 생성 함수의 호출 형식이 맞지 않습니다. 앱을 재시작해 최신 코드를 불러와 주세요.')
                            break
                        except Exception:
                            parts.append(f'{n}번 기준 {index + 1}번째 문제를 생성하지 못했습니다. API 연결·설정 또는 응답 시간 한도를 확인해 주세요.')
                            break
                    if n not in used:
                        used.append(n)
                    lines = [f'**{n}번 유형의 독립적인 신규 문제 {index + 1}/{count}**', generated.get('passage', ''), generated['question'], *[f'{i}. {v}' for i, v in enumerate(generated['choices'], 1)], f"정답: {generated['correct_answer']}번", f"해설: {generated['explanation']}"]
                    if source.get('listening_without_script'):
                        lines.insert(1, '원본 듣기 대본은 제공되지 않았습니다. 아래 대본은 새로 만든 독립적인 대본입니다.')
                    parts.append('\n\n'.join(v for v in lines if v))
            elif item['detail'] == 'explain':
                sources.append({**source, 'answer': key['answer'], 'points': key['points']})
            else:
                warning = '\n\n주의: 추출이 불완전할 수 있습니다. PDF 원본을 함께 확인하세요.' if source.get('extraction_status') == 'partial' else ''
                parts.append(f"**{n}번 문제 (PDF {source['page']}페이지)**\n\n{source['text']}\n\n정답: {key['answer']}번{warning}")
        if sources:
            if not state.get('api_key'):
                parts.append('문항 해설에는 OpenAI API 키 설정이 필요합니다.')
            else:
                parts.append(explain_question(state['query'], sources, state['api_key'], state['model'], timeout=min(14.0, _remaining(state)), problem_pdf=state.get('problem_pdf')))
        return '\n\n'.join(parts)
    return CLARIFY_TEXT


def answer_node(state):
    context, answers = dict(state['context']), []
    for item in state['plan']:
        try:
            answers.append(_execute(item, state, context))
        except TimeoutError:
            answers.append('응답 시간 한도에 도달했습니다. 범위를 좁혀 다시 질문해 주세요. 재분석은 필요하지 않습니다.')
        except Exception:
            answers.append('외부 API 또는 자료 처리에 실패했습니다. API 설정·연결 또는 추출 상태를 확인해 주세요. 재분석은 필요하지 않습니다.')
    return {'answer': '\n\n'.join(answers), 'context': context}


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node('router', router)
    builder.add_node('answer', answer_node)
    builder.add_edge(START, 'router')
    builder.add_edge('router', 'answer')
    builder.add_edge('answer', END)
    return builder.compile()


CHAT_GRAPH = build_graph()  # No checkpointer, report or shared user memory.


def run_chat(query: str, report: dict, history: list[dict], api_key: str = '', model: str = DEFAULT_MODEL, cached_question: dict | None = None, problem_pdf: bytes | None = None, last_student: str | None = None, context: dict | None = None) -> tuple[str, str | None]:
    session = dict(context or {})
    if last_student and 'last_students' not in session:
        session['last_students'] = [last_student]
    output = CHAT_GRAPH.invoke({'query': query, 'report': report, 'history': history[-12:], 'api_key': api_key, 'model': model, 'cached_question': cached_question, 'problem_pdf': problem_pdf, 'context': session, 'deadline': time.monotonic() + 18.0})
    if context is not None:
        context.clear()
        context.update(output['context'])
    names = output['context'].get('last_students', [])
    return output['answer'], names[0] if len(names) == 1 else None
