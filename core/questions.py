"""Coordinate-aware question regions, shared passages and explicit completeness."""
from __future__ import annotations
from collections import defaultdict
import re
import pymupdf

QUESTION_START = re.compile(r'^\s*(\d{1,3})\s*[.．)]\s*(.*)$')
GROUP_START = re.compile(r'\[\s*(\d{1,3})\s*[~～∼\-–]\s*(\d{1,3})\s*\]')
OPTIONS = re.compile(r'[①②③④⑤❶❷❸❹❺]|(?<![A-Za-z])[A-E][.)](?=\s)')


def _lines(page, rect):
    lines = []
    for block in page.get_text('dict', clip=rect)['blocks']:
        for line in block.get('lines', []):
            text = ''.join(s['text'] for s in line['spans']).strip()
            box = list(line['bbox'])
            if not text or '이 문제지에 관한 저작권' in text:
                continue
            if re.fullmatch(r'\d+|홀수형|짝수형|제\s*\d+\s*교시', text) and (box[1] < 35 or box[3] > page.rect.height - 35):
                continue
            lines.append({'text': text, 'bbox': box})
    return sorted(lines, key=lambda line: (round(line['bbox'][1] / 3), line['bbox'][0]))


def parse_question_pdf(content: bytes, expected_numbers: list[int]) -> tuple[dict[int, dict], list[int]]:
    if not content or len(content) > 35_000_000:
        raise ValueError('문제 PDF가 비어 있거나 35MB를 초과했습니다.')
    questions, groups = {}, {}
    active = None
    current_group = None
    try:
        pdf = pymupdf.open(stream=content, filetype='pdf')
    except Exception as exc:
        raise ValueError('문제 PDF를 열 수 없습니다. 파일 형식/손상을 확인해 주세요.') from exc
    with pdf:
        if pdf.needs_pass or not 1 <= len(pdf) <= 80:
            raise ValueError('문제 PDF는 암호가 없는 1~80페이지여야 합니다.')
        for page_i, page in enumerate(pdf):
            width, height = page.rect.width, page.rect.height
            whole = _lines(page, page.rect)
            left = [l for l in whole if l['bbox'][0] < width * .4]
            right = [l for l in whole if l['bbox'][0] >= width / 2]
            crossing = [l for l in whole if l['bbox'][0] < width / 2 - 10 and l['bbox'][2] > width / 2 + 10]
            right_heading = any(QUESTION_START.match(l['text']) for l in right)
            two_columns = right_heading or (len(left) >= 4 and len(right) >= 4 and len(crossing) <= 1)
            columns = [(0, width / 2, '좌'), (width / 2, width, '우')] if two_columns else [(0, width, '전체')]
            for x0, x1, column in columns:
                lines = _lines(page, pymupdf.Rect(x0, 0, x1, height))
                events = []
                for i, line in enumerate(lines):
                    tag, heading = GROUP_START.search(line['text']), QUESTION_START.match(line['text'])
                    if tag and 1 <= int(tag[1]) <= int(tag[2]) <= 150:
                        events.append((i, 'group', (int(tag[1]), int(tag[2]))))
                    elif heading:
                        events.append((i, 'q', int(heading[1])))

                def append_chunk(start, end, target):
                    if target is None or start >= end:
                        return
                    mode, number = target
                    store = groups if mode == 'group' else questions
                    if number not in store:
                        return
                    chunk = '\n'.join(l['text'] for l in lines[start:end])
                    y0 = max(0, min(l['bbox'][1] for l in lines[start:end]) - 3)
                    y1 = lines[end]['bbox'][1] - 3 if end < len(lines) else height - 15
                    region = {'page': page_i + 1, 'bbox': [x0, y0, x1, max(y0 + 1, y1)]}
                    store[number]['parts'].append(chunk)
                    store[number]['regions'].append(region)

                append_chunk(0, events[0][0] if events else len(lines), active)
                for index, (start, mode, number) in enumerate(events):
                    end = events[index + 1][0] if index + 1 < len(events) else len(lines)
                    if mode == 'group':
                        current_group = number
                        groups.setdefault(number, {'parts': [], 'regions': []})
                        active = ('group', number)
                    else:
                        if number not in expected_numbers:
                            active = None
                            continue
                        if current_group and not current_group[0] <= number <= current_group[1]:
                            current_group = None
                        if number in questions:
                            questions[number]['issues'].append('duplicate_heading')
                        else:
                            questions[number] = {'number': number, 'page': page_i + 1, 'column': column,
                                'parts': [], 'regions': [], 'issues': [], 'shared_group': current_group}
                        active = ('q', number)
                    append_chunk(start, end, active)
        for number, q in questions.items():
            own = '\n'.join(q.pop('parts'))
            group = groups.get(q['shared_group'])
            shared = '\n'.join(dict.fromkeys(group['parts'])) if group else ''
            q['shared_text'] = shared
            q['text'] = '\n'.join(p for p in (shared, own) if p)
            if group:
                q['regions'] = group['regions'] + q['regions']
            choices = OPTIONS.findall(own)
            q['choice_count'] = len(choices)
            q['has_visual'] = bool(re.search(r'그림|도표|그래프|표를|graph|chart|table\s+(?:below|above)|illustration', q['text'], re.I))
            for region in q['regions']:
                page = pdf[region['page'] - 1]
                rect = pymupdf.Rect(region['bbox'])
                image_boxes = [pymupdf.Rect(im['bbox']) for im in page.get_image_info()]
                q['has_visual'] |= any((rect & box).get_area() > 25 for box in image_boxes)
                # Vector charts are also visual, even when no raster is embedded.
                q['has_visual'] |= any((rect & drawing['rect']).get_area() > 100 for drawing in page.get_drawings())
            q['listening_without_script'] = bool(re.search(r'듣고|대화를\s*듣|listen|you\s+will\s+hear', own, re.I) and not shared)
            if len(q['text'].strip()) < 40:
                q['issues'].append('short_text')
            if len(q['text']) > 20000 or len(q['regions']) > 5:
                q['issues'].append('oversized_or_ambiguous_region')
            if len(choices) != 5:
                q['issues'].append('choices_unverified')
            if q['shared_group'] and not shared.strip():
                q['issues'].append('missing_shared_passage')
            q['extraction_status'] = ('partial' if q['issues'] else 'visual_pending' if q['has_visual'] else 'complete')
            q['confidence'] = .45 if q['issues'] else .8 if q['has_visual'] else .95
    return questions, sorted(set(expected_numbers) - questions.keys())
