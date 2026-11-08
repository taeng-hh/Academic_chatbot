"""Read answer keys from the uploaded PDF. Never store/hardcode answer keys."""
from __future__ import annotations

import io
import re

import pymupdf

CHOICE_SYMBOLS = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5,
                  "❶": 1, "❷": 2, "❸": 3, "❹": 4, "❺": 5}


def choice_number(value: object) -> int | None:
    """Convert Korean circled/ASCII choices; None denotes an empty OMR bubble."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"유효하지 않은 선택지: {value}")
    if isinstance(value, float) and value != value:  # NaN
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "미표기", "무응답", "-"}:
        return None
    text = re.sub(r"\s+", "", text)
    if text in CHOICE_SYMBOLS:
        return CHOICE_SYMBOLS[text]
    if re.fullmatch(r"[1-5](?:\.0+)?", text):
        return int(float(text))
    raise ValueError(f"1~5 범위가 아닌 응답값: {value!r}")


def _read_rows(rows: list[list[object]]) -> dict[int, dict[str, int]]:
    found: dict[int, dict[str, int]] = {}
    aliases = {'number': {'문항', '문항번호', '번호', 'question', 'no', 'no.'},
               'answer': {'정답', '답', 'answer'}, 'points': {'배점', '점수', 'points', 'score'}}
    layout = None
    for row in rows:
        labels = [str(c or '').lower().replace(' ', '').replace('\n', '') for c in row]
        roles = {role: [i for i, c in enumerate(labels) if c in words] for role, words in aliases.items()}
        if all(roles.values()) and len({len(v) for v in roles.values()}) == 1:
            layout = list(zip(roles['number'], roles['answer'], roles['points']))
            continue
        if len(row) < 3:
            continue
        for i, a_i, p_i in (layout or [(i, i+1, i+2) for i in range(0, len(row)-2, 3)]):
            if max(i, a_i, p_i) >= len(row):
                raise ValueError('정답표의 열 구성이 행마다 다릅니다.')
            no_text = str(row[i] or "").strip()
            if not re.fullmatch(r"\d{1,3}", no_text):
                continue
            number = int(no_text)
            answer = choice_number(row[a_i])
            point_text = str(row[p_i] or "").strip()
            if answer is None or not re.fullmatch(r"\d+(?:\.\d+)?", point_text):
                raise ValueError(f"{number}번: 정답 또는 배점을 추출하지 못했습니다.")
            points = float(point_text)
            points = int(points) if points.is_integer() else points
            if points <= 0 or points > 10000:
                raise ValueError(f"{number}번: 배점이 비정상적입니다: {points}")
            if number in found:
                raise ValueError(f"정답 PDF에 {number}번 문항이 중복되었습니다.")
            found[number] = {"answer": answer, "points": points}
    return found


def parse_answer_pdf(content: bytes) -> dict[int, dict[str, int]]:
    """Extract question-number/answer/points from table on uploaded PDF.

    Falls back to pdfplumber for borderless tables. No hardcoded answers.
    """
    if not content or len(content) > 25_000_000:
        raise ValueError("정답 PDF가 비어 있거나 25MB를 초과했습니다.")
    records = {}
    fallback_rows = []
    try:
        with pymupdf.open(stream=content, filetype="pdf") as doc:
            if doc.needs_pass or not 1 <= len(doc) <= 50:
                raise ValueError("정답 PDF는 50페이지 이하여야 합니다.")
            for page in doc:
                try:
                    tables = page.find_tables().tables
                except (RuntimeError, AttributeError, ValueError):
                    tables = []
                for table in tables:
                    parsed = _read_rows(table.extract())
                    overlap = set(records) & set(parsed)
                    if overlap:
                        raise ValueError(f"문항 번호 중복: {sorted(overlap)}")
                    records.update(parsed)
                # Borderless rows, only on pages explicitly labelled as a key.
                if re.search(r'정답|배점|answer|points', page.get_text(), re.I):
                    lines = {}
                    for word in page.get_text('words'):
                        lines.setdefault(round(word[1] / 3), []).append(word)
                    fallback_rows.extend([[w[4] for w in sorted(words, key=lambda w: w[0])]
                                          for words in lines.values()])
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"정답 PDF 처리 실패: {exc}") from exc

    if not records:
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    for rows in page.extract_tables() or []:
                        parsed = _read_rows(rows)
                        if set(records) & set(parsed):
                            raise ValueError("정답 번호가 중복되었습니다.")
                        records.update(parsed)
        except ImportError as exc:
            raise ValueError("표 추출 실패: pdfplumber를 설치하거나 정답 PDF를 확인하세요.") from exc

    if not records and fallback_rows:
        records = _read_rows(fallback_rows)

    if not records:
        raise ValueError("PDF에서 문항·정답·배점으로 된 표를 찾지 못했습니다.")
    numbers = sorted(records)
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError("정답 번호에 누락 또는 비연속 번호가 있습니다.")
    if not (1 <= len(records) <= 150):
        raise ValueError("문항 수가 허용 범위(1~150)를 벗어났습니다.")
    return dict(sorted(records.items()))
