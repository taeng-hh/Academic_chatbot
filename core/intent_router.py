"""Compositional intent parsing and validation; never asks a model for scores."""
from __future__ import annotations

import re
from typing import Any

KINDS = ['summary', 'lowest', 'highest', 'student', 'students', 'compare', 'study', 'missed', 'questions',
         'question', 'generate', 'search', 'general', 'features', 'unsupported', 'security', 'clarify']
METRICS = ['mean', 'median', 'std_population', 'variance_population', 'student_count', 'question_count', 'max_score', 'score_distribution']
DETAILS = ['score', 'wrong', 'blank', 'correct', 'details', 'text', 'answer', 'stats', 'explain']
ACTION_PROPERTIES = {
    'kind': {'type': 'string', 'enum': KINDS},
    'students': {'type': 'array', 'items': {'type': 'string'}},
    'questions': {'type': 'array', 'items': {'type': 'integer'}},
    'metrics': {'type': 'array', 'items': {'type': 'string', 'enum': METRICS}},
    'detail': {'type': 'string', 'enum': DETAILS},
    'limit': {'type': ['integer', 'null']},
    'lower': {'type': ['number', 'null']}, 'upper': {'type': ['number', 'null']},
    'lower_inclusive': {'type': 'boolean'}, 'upper_inclusive': {'type': 'boolean'},
    'relative_to': {'type': 'string', 'enum': ['none', 'above_mean', 'at_least_mean', 'below_mean', 'above_median', 'below_median']},
    'choice': {'type': ['integer', 'null']},
    'status': {'type': 'string', 'enum': ['all', 'correct', 'wrong', 'blank']},
    'rank': {'type': ['integer', 'null']}, 'rank_end': {'type': ['integer', 'null']},
    'rank_mode': {'type': 'string', 'enum': ['competition', 'dense']},
    'rank_from': {'type': 'string', 'enum': ['top', 'bottom']},
    'order_by': {'type': 'string', 'enum': ['score', 'name', 'number', 'miss_count', 'miss_rate', 'points']},
    'descending': {'type': 'boolean'}, 'include_ties': {'type': 'boolean'},
    'keyword': {'type': 'string'},
}
INTENT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'actions': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
                    'properties': ACTION_PROPERTIES, 'required': list(ACTION_PROPERTIES)}},
        'confidence': {'type': 'number'}, 'requires_clarification': {'type': 'boolean'},
    }, 'required': ['actions', 'confidence', 'requires_clarification'],
}
GREET = re.compile(r'(?:안녕(?:하세요|하십니까)?|하잉|하이|ㅎㅇ|반가워(?:요)?|hello|hi|hey)[\s.!?~ㅋㅎ🙂]*', re.I)
THANKS = re.compile(r'(?:고마워(?:요)?|감사(?:합니다|해요)?|땡큐|thanks|thank you)[\s.!?~]*', re.I)
FEATURES = re.compile(r'(?:어떤|무슨).*(?:기능|지원|일을\s*할)|기능.*(?:알려|설명)|할\s*수\s*있(?:는|어)')
SECRET = re.compile(r'(?:api\s*키|api\s*key|비밀\s*키|환경\s*변수|시스템\s*(?:프롬프트|지침|메시지)|개발자\s*지침|내부\s*(?:도구|tool|메시지|설정)|어떤\s*tool\s*연결)', re.I)
DISCLOSE = re.compile(r'출력|공개|보여|알려|노출|읽어|reveal|disclose|연결되어', re.I)
CONCEPT = re.compile(r'무엇|뭐야|개념|뜻|예방|방어|정의')
QUESTION = re.compile(r'(?<!\d)(\d{1,3})\s*번(?!\s*학생)')
EXAM = re.compile(r'학생|성적|점수|시험|모의고사|OMR|오답|정답|문항|문제지|답지|답안|배점|지문|\d+\s*점|석차|등수', re.I)
PRONOUN = re.compile(r'그\s*(?:학생|친구|사람|문제|문항)|이\s*학생|해당\s*(?:학생|문항|문제)')
UNAVAILABLE = re.compile(r'출석|출결|참석|결석|지각|조퇴|수업\s*(?:에\s*)?참여|수업\s*(?:에\s*)?(?:왔|나왔)|주소|전화번호|연락처|생년월일|성별|학부모|보호자|숙제\s*(?:제출|했)|과제\s*제출')


def action(kind: str, **kwargs) -> dict:
    return {'kind': kind, 'students': [], 'questions': [], 'metrics': [],
            'detail': 'score', 'limit': None, 'lower': None, 'upper': None, 'keyword': '',
            'lower_inclusive': True, 'upper_inclusive': True, 'relative_to': 'none', 'choice': None, 'status': 'all',
            'rank': None, 'rank_end': None, 'rank_mode': 'competition', 'rank_from': 'top',
            'order_by': 'score', 'descending': True, 'include_ties': True, **kwargs}


def validate_plan(value: Any) -> list[dict]:
    """Validate model output independently from provider schema enforcement."""
    if not isinstance(value, dict) or set(value) != {'actions', 'confidence', 'requires_clarification'}:
        raise ValueError('의도 응답 구조 오류')
    confidence = value['confidence']
    if type(confidence) not in (int, float) or not 0 <= confidence <= 1 or type(value['requires_clarification']) is not bool:
        raise ValueError('의도 신뢰도 오류')
    if not isinstance(value['actions'], list) or not 1 <= len(value['actions']) <= 8:
        raise ValueError('의도 개수 오류')
    for item in value['actions']:
        if not isinstance(item, dict) or set(item) != set(ACTION_PROPERTIES):
            raise ValueError('의도 필드 오류')
        if item['kind'] not in KINDS or item['detail'] not in DETAILS:
            raise ValueError('허용되지 않은 작업')
        if type(item['lower_inclusive']) is not bool or type(item['upper_inclusive']) is not bool or item['relative_to'] not in ACTION_PROPERTIES['relative_to']['enum']:
            raise ValueError('조건 비교 형식 오류')
        if item['status'] not in {'all', 'correct', 'wrong', 'blank'} or (item['choice'] is not None and (type(item['choice']) is not int or not 1 <= item['choice'] <= 5)):
            raise ValueError('응답 선택 조건 오류')
        for field in ('rank', 'rank_end'):
            if item[field] is not None and (type(item[field]) is not int or not 1 <= item[field] <= 5000):
                raise ValueError('석차 조건 오류')
        if item['rank_end'] is not None and (item['rank'] is None or item['rank_end'] < item['rank']):
            raise ValueError('석차 범위 오류')
        for field in ('rank_mode', 'rank_from', 'order_by'):
            if item[field] not in ACTION_PROPERTIES[field]['enum']:
                raise ValueError('정렬 방식 오류')
        if type(item['descending']) is not bool or type(item['include_ties']) is not bool:
            raise ValueError('정렬 조건 형식 오류')
        for field, element_type in [('students', str), ('questions', int), ('metrics', str)]:
            if not isinstance(item[field], list) or len(item[field]) > 150 or any(type(v) is not element_type for v in item[field]):
                raise ValueError('조회 조건 형식 오류')
        if any(m not in METRICS for m in item['metrics']) or any(not 1 <= n <= 150 for n in item['questions']):
            raise ValueError('조회 범위 오류')
        if item['limit'] is not None and (type(item['limit']) is not int or not 1 <= item['limit'] <= 5000):
            raise ValueError('조회 개수 오류')
        if any(item[f] is not None and (type(item[f]) not in (int, float) or not -10000 <= item[f] <= 10000) for f in ('lower', 'upper')):
            raise ValueError('점수 조건 오류')
        if not isinstance(item['keyword'], str) or len(item['keyword']) > 2000 or any(len(s) > 200 for s in item['students']):
            raise ValueError('문자열 조건 오류')
    if value['requires_clarification'] or confidence < .7:
        return [action('clarify')]
    return value['actions']


def resolve_students(refs: list[str], report: dict) -> tuple[list[dict], str | None]:
    selected = []
    for ref in refs:
        ref = re.sub(r'\s+', '', ref)
        exact = [s for s in report['students'] if re.sub(r'\s+', '', s['name']) == ref]
        if exact:
            matches = exact
        else:
            def alias(name):
                name = re.sub(r'\s+', '', name)
                return re.sub(r'(?:번)?학생$', '', name)
            target = re.sub(r'(?:번)?(?:학생)?$', '', ref)
            matches = [s for s in report['students'] if alias(s['name']) == target]
        if not matches:
            return [], f'학생 식별자 “{ref}”는 현재 업로드된 OMR에서 확인할 수 없습니다.'
        if len(matches) > 1:
            return [], f'“{ref}”에 해당하는 학생이 여러 명입니다. 정확한 학생명/ID를 알려주세요.'
        if matches[0] not in selected:
            selected.append(matches[0])
    return selected, None


def _mentioned_students(q: str, report: dict) -> list[str]:
    refs, spans = [], []
    def pattern(identifier):
        suffix = r'(?=$|[\s,.?!]|은|는|이|가|의|을|를|도|랑|과|와)' if identifier.isdigit() else r'(?=$|[\s,.?!]|학생|번|은|는|이|가|의|을|를|도|랑|과|와)'
        return r'(?<!\w)' + re.escape(identifier) + suffix
    # First honor full identifiers. Alias matches inside these spans must not
    # turn an exact record into a second, different student's identifier.
    for s in report['students']:
        name = s['name']
        for match in re.finditer(pattern(name), q):
            refs.append(name)
            spans.append(match.span())
    def occupied(match):
        return any(start <= match.start() < end for start, end in spans)
    for s in report['students']:
        alias = re.sub(r'(?:번)?\s*학생$', '', s['name'])
        if alias and alias != s['name']:
            for match in re.finditer(pattern(alias), q):
                if not occupied(match):
                    refs.append(alias + ('번학생' if q[match.end():].startswith('번') else ''))
    for match in re.finditer(r'(?<!\w)([\w-]+)\s*(?:번\s*)?학생', q):
        if occupied(match):
            continue
        ref = match.group(1)
        if re.search(r'\d+\s*(?:등|위)$|번째$|^(?:받은|기록한|해당하는|차지한)$', ref):
            continue
        if ref not in {'그', '이', '해당', '모든', '전체', '각', '최하위', '최상위', '높은', '낮은', '성적', '점수', '최저', '최고', '최저점', '최고점', '꼴찌', '동점', '없는', '저조한'}:
            refs.append(ref)
    if re.search(r'점수|몇\s*점|총점|성적', q):
        for match in QUESTION.finditer(q):
            if occupied(match):
                continue
            n = int(match.group(1))
            if not re.search(r'문항|문제|배점|정답', q):
                refs.append(match.group(1) + '번학생')
    return list(dict.fromkeys(refs))


def rule_plan(q: str, report: dict, context: dict) -> list[dict] | None:
    """Only return a plan when all detected slots can be answered safely."""
    q = q.strip()
    concept_only = bool(re.search(r'개념|정의|무엇|뭐야|뭔지|관리|보관|예방|방어', q) and not re.search(r'전체|전문|실제|너의|비공개|원문|출력|보여|reveal|disclose', q, re.I))
    if SECRET.search(q) and DISCLOSE.search(q) and not concept_only:
        return [action('security')]
    if GREET.fullmatch(q) or THANKS.fullmatch(q):
        return [action('general')]
    if FEATURES.search(q) and not re.search(r'평균|중앙|몇\s*점|오답률', q):
        return [action('features')]
    refs = _mentioned_students(q, report)
    source_question = bool(re.search(r'지문|문제지|본문|문항|보기|단어|뜻|개념|번역|영어로', q))
    unavailable = bool(UNAVAILABLE.search(q) and (refs or '학생' in q or PRONOUN.search(q)) and not source_question)
    # Attendance/privacy is absent regardless of which student was referenced.
    if unavailable and not re.search(r'평균|중앙|표준|분산|점수|성적|오답|정답|배점|석차|등수', q):
        return [action('unsupported')]
    previous_exam = re.search(r'(?:지난|이전|지난번|저번|작년).{0,12}(?:시험|모의고사|성적|점수)', q)
    if previous_exam and (refs or EXAM.search(q)):
        return [action('unsupported')]
    if re.search(r'그\s*(?:학생|친구|사람)|이\s*학생', q):
        refs = context.get('last_students', [])
        if len(refs) != 1:
            if re.search(r'\d+\s*(?:등|위)|번째|최저|최고|최하위|최상위|꼴찌', q):
                return None
            return [action('clarify')]
    numbers = [] if refs and not re.search(r'문항|문제|정답|오답|지문|보기|배점', q) else [int(n) for n in QUESTION.findall(q)]
    if re.search(r'그\s*(?:문제|문항)|해당\s*(?:문제|문항)', q) and not numbers:
        numbers = context.get('last_questions', [])
        if len(numbers) != 1:
            return [action('clarify')]
    generate = bool(re.search(r'만들|출제|생성', q) and (numbers or re.search(r'문제|문항|유형', q)))
    rank_match = re.search(r'(?<!\d)(\d+)\s*(?:등(?!급)|위|번째)', q)
    ordinal = re.search(r'(첫|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*번째|(?:첫째|둘째|셋째|넷째)', q)
    rank_number = int(rank_match[1]) if rank_match else None
    if rank_number is None and ordinal:
        rank_number = {'첫': 1, '두': 2, '세': 3, '네': 4, '다섯': 5, '여섯': 6, '일곱': 7, '여덟': 8, '아홉': 9, '열': 10,
                       '첫째': 1, '둘째': 2, '셋째': 3, '넷째': 4}.get(ordinal[1] or ordinal[0])
    ranking = rank_number is not None and not numbers and not refs and not re.search(r'문항|문제|보기', q)
    if rank_number is not None and re.search(r'문항|문제|보기', q):
        return None
    if rank_number is not None and refs and not numbers and not any(resolve_students([r], report)[0] for r in refs):
        return None
    top_count = re.search(r'(상위|하위)\s*(\d+)\s*(?:명|학생)', q)
    if generate and refs and re.search(r'점수|성적|몇\s*점', q):
        return None
    # Explanations, comparisons and conditions have subtleties; use schema
    # interpretation instead of adding another string-specific answer branch.
    if re.search(r'비교|차이|보다|이상|이하|초과|미만|왜|근거|해석|번역|해설|추천|계획|어휘|문법|주제|취약|약점|설명|공부|분석|백분|등급|고른|선택한|맞힌|맞춘', q):
        return None
    metrics = []
    for metric, pattern in [('mean', r'평균'), ('median', r'중앙\s*값|중심값'),
                            ('std_population', r'표준\s*편차'), ('variance_population', r'분산'),
                            ('student_count', r'학생\s*수|몇\s*명'), ('question_count', r'문항\s*수'),
                            ('max_score', r'만점|총\s*배점')]:
        if re.search(pattern, q):
            metrics.append(metric)
    actions = []
    if unavailable:
        actions.append(action('unsupported'))
    if re.search(r'(?:전체|성적|결과).*(?:요약|분포)|전체\s*통계', q):
        metrics = METRICS[:]
    if (ranking or top_count) and '몇' in q and not re.search(r'전체\s*학생\s*수', q):
        metrics = [m for m in metrics if m != 'student_count']
    if metrics and not (refs and '몇' in q):
        actions.append(action('summary', metrics=metrics))
    if ranking:
        actions.append(action('students', rank=rank_number,
            rank_mode='dense' if re.search(r'(?:번째|째).*?(?:높|낮).*점수|점수.*(?:번째|째)', q) else 'competition',
            rank_from='bottom' if re.search(r'낮|뒤에서|하위|아래|밑에서', q) else 'top'))
    elif top_count and not refs and not re.search(r'문항|문제|보기', q):
        actions.append(action('students', limit=int(top_count[2]), descending=top_count[1] == '상위'))
    if not (ranking or top_count) and re.search(r'최저|최하위|꼴찌|(?:제일|가장).*(?:못\s*본|저조|낮)', q) and not re.search(r'오답|정답률|문항|문제', q):
        actions.append(action('lowest'))
    if not (ranking or top_count) and re.search(r'최고\s*점|최상위|수석|(?<!\d)1\s*등(?!급)|(?:가장|제일).*(?:높|잘\s*본)', q) and not re.search(r'오답|정답률|문항|문제', q):
        actions.append(action('highest'))
    missed = bool(re.search(r'오답\s*(?:률|율|순위|상위)|최다\s*오답|많이\s*틀', q))
    if ranking or top_count:
        pass
    elif generate:
        quantity = re.search(r'(\d+)\s*(?:개|문항|문제)', q)
        actions.append(action('generate', students=refs, questions=numbers, limit=int(quantity[1]) if quantity else 1))
    elif refs and (not unavailable or re.search(r'점수|성적|몇\s*점|오답|정답', q)):
        detail = 'wrong' if re.search(r'오답|틀', q) else 'blank' if re.search(r'미응답|무응답|공란', q) else 'correct' if re.search(r'맞|정답', q) else 'score'
        if numbers:
            detail = 'details'
        actions.append(action('student', students=refs, questions=numbers, detail=detail))
    elif numbers:
        detail = 'stats' if re.search(r'오답|응답|고른|선택한|맞힌|맞춘|틀린|몇\s*명', q) else 'answer' if '정답' in q else 'text'
        actions.append(action('question', questions=numbers, detail=detail))
    elif missed:
        count = re.search(r'(\d+)\s*(?:개|문항|문제)', q)
        if re.search(r'어떤\s*문제|무슨\s*문제|본문|지문', q) and report.get('question_stats'):
            top = min(report['question_stats'], key=lambda row: (-row['miss_count'], row['number']))
            actions.append(action('question', questions=[top['number']], detail='text'))
        else:
            actions.append(action('missed', limit=int(count.group(1)) if count else 1))
    elif re.search(r'석차|등수|순위|학생.*목록', q):
        actions.append(action('students'))
    if actions:
        return actions
    if PRONOUN.search(q):
        return [action('clarify')]
    return None
