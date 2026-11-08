"""Build a self-contained Excel file with summary, grading matrix, and item analysis."""
from __future__ import annotations

import io
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.comments import Comment
from openpyxl.utils import get_column_letter
from .report_tools import query_exam


def safe_excel_text(value: str) -> str:
    """Prevent untrusted student names from being interpreted as Excel formulas."""
    if value and (value[:1] in ("\t", "\r", "\n") or value.lstrip().startswith(("=", "+", "-", "@"))):
        return "'" + value
    return value


def export_results(report: dict) -> bytes:
    summary = report["summary"]
    numbers = sorted(report["answer_key"])
    most_missed = sorted(report['question_stats'], key=lambda item: (-item['miss_count'], item['number']))[:3]
    wb = Workbook()
    ws = wb.active
    ws.title = "전체 성적 개요"
    stats = [
        ("중앙값", summary["median"]),
        ("평균", summary["mean"]),
        ("표준편차(모집단)", summary["std_population"]),
        ('가장 많이 틀린 문항 번호 3개', ', '.join(f"{item['number']}번" for item in most_missed)),
        ("총 학생 수", summary["student_count"]),
        ("총 문항 수", summary["question_count"]),
        ("총 배점", summary["max_score"]),
        ("최저 점수", summary["lowest_score"]),
        ("최저 점수 학생", safe_excel_text(", ".join(summary["lowest_students"]))),
    ]
    ws.append(["항목", "값"])
    for row in stats:
        ws.append(row)
        if row[0] in {'평균', '중앙값', '표준편차(모집단)'}:
            ws.cell(ws.max_row, 2).number_format = '0.00'
    ws.append([])
    ws.append(["최다 오답 순위", "문항 번호", "오답 인원", "오답률"])
    missed_header_row = ws.max_row
    for rank, item in enumerate(most_missed, 1):
        ws.append([rank, item["number"], item["miss_count"], item["miss_rate"]])
        ws.cell(ws.max_row, 4).number_format = "0.0%"
    ws.append([])
    ws.append(['정답 여부 표기', 'O = 정답 / X = 오답(미응답 포함)'])
    ws.append(['표준편차 기준', '전체 학생을 모집단으로 계산'])
    ws.append(['오답 동률 기준', '오답 인원 내림차순 → 문항 번호 오름차순'])
    if len(numbers) < 3:
        ws.append(['오답 상위 문항 안내', '전체 문항이 3개 미만이므로 존재하는 문항만 표시'])

    matrix = wb.create_sheet("학생별 채점 결과")
    matrix.append(["학생명", "총점", *[f"{n}번 정답 여부" for n in numbers]])
    matrix['A1'].comment = Comment('한 행은 학생 한 명입니다. 총점과 모든 문항의 정답 여부를 표시합니다. O = 정답, X = 오답(미응답 포함). 전체 평균·중앙값·표준편차·오답 상위 3문항은 같은 파일 하단의 “전체 성적 개요” 시트에 있습니다.', '평가 도우미')
    for r in report["students"]:
        matrix.append([safe_excel_text(r["name"]), r["score"],
                       *["O" if r["correct"][n] else "X" for n in numbers]])
    for row in matrix.iter_rows(min_row=2, min_col=3):
        for cell in row:
            cell.font = Font(color='146C43' if cell.value == 'O' else 'B42318', bold=True)
            cell.fill = PatternFill('solid', fgColor='EAF6EE' if cell.value == 'O' else 'FDECEC')
            cell.alignment = Alignment(horizontal='center')
            number = numbers[cell.column - 3]
            student = report['students'][cell.row - 2]
            submitted = student['answers'][number]
            key = report['answer_key'][number]
            cell.comment = Comment(
                f"{'정답' if cell.value == 'O' else '오답'}\n"
                f"제출 답: {submitted if submitted is not None else '미응답'}\n"
                f"정답: {key['answer']}번\n배점: {key['points']}점", '평가 도우미')
    items = wb.create_sheet("문항별 분석")
    items.append(["문항", "정답", "배점", "정답 인원", "오답 인원", "오답률"])
    for r in report["question_stats"]:
        items.append([r["number"], r["answer"], r["points"], r["correct_count"],
                      r["miss_count"], r["miss_rate"]])
        items.cell(items.max_row, 6).number_format = "0.0%"
    ranking = wb.create_sheet('학생별 석차 및 답안')
    ranking.append(['학생명', '전체 석차', '서로 다른 점수 순서', '총점', '정답 문항 수', '오답 문항 수', '미응답 문항 수', '감점',
                    *[f'Q{n:02d} 제출 답' for n in numbers]])
    for r in query_exam(report, 'student')['students']:
        ranking.append([safe_excel_text(r['name']), r['rank'], r['score_rank'], r['score'], r['correct_count'],
                        len(r['wrong_numbers']), len(r['blank_numbers']), r['lost_points'],
                        *[a['submitted_answer'] if a['submitted_answer'] is not None else '미응답' for a in r['responses']]])
    for sheet in wb.worksheets:
        sheet.freeze_panes = "B2"
        if sheet != ws:
            sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.fill = PatternFill("solid", fgColor="15324D")
            cell.font = Font(bold=True, color="FFFFFF")
            cell.alignment = Alignment(horizontal="center")
        sheet.column_dimensions["A"].width = 20
        sheet.column_dimensions["B"].width = 24
        for c in range(3, sheet.max_column+1):
            sheet.column_dimensions[get_column_letter(c)].width = 11
    ws.column_dimensions['A'].width = 34
    ws.column_dimensions['B'].width = 54
    ws.sheet_properties.tabColor = '2563EB'
    for row in ws.iter_rows(min_row=2, max_row=5, max_col=2):
        for cell in row:
            cell.fill = PatternFill('solid', fgColor='EAF0FC')
            cell.font = Font(bold=True, color='15324D')
    for cell in ws[missed_header_row]:
        cell.fill = PatternFill('solid', fgColor='15324D')
        cell.font = Font(bold=True, color='FFFFFF')
        cell.alignment = Alignment(horizontal='center')
    matrix.freeze_panes = 'C2'
    matrix.column_dimensions['B'].width = 14
    for column in range(3, matrix.max_column + 1):
        matrix.column_dimensions[get_column_letter(column)].width = 17
    matrix.sheet_properties.tabColor = '247B67'
    # Make the requested student totals and O/X visible immediately on opening.
    wb.move_sheet(matrix, offset=-1)
    wb.active = 0
    for sheet in wb.worksheets:
        sheet.sheet_view.tabSelected = sheet is matrix
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()
