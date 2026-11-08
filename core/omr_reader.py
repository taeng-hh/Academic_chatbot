"""Read one student per row, student column and Q01..Q45 type columns."""
from __future__ import annotations

import io
import re
import zipfile
import pandas as pd
from .pdf_reader import choice_number


def parse_omr_excel(content: bytes, question_numbers: list[int]) -> list[dict]:
    if not content or len(content) > 15_000_000:
        raise ValueError("OMR Excel이 비어 있거나 15MB를 초과했습니다.")
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if sum(f.file_size for f in archive.infolist()) > 150_000_000:
                raise ValueError('OMR Excel의 압축 해제 크기가 너무 큽니다.')
    except zipfile.BadZipFile as exc:
        raise ValueError('올바른 XLSX 파일이 아닙니다.') from exc
    with pd.ExcelFile(io.BytesIO(content), engine="openpyxl") as book:
        sheet = "OMR_응답" if "OMR_응답" in book.sheet_names else book.sheet_names[0]
        df = pd.read_excel(book, sheet_name=sheet, dtype=object)
    df.columns = [str(col).strip() for col in df.columns]
    if len(set(df.columns)) != len(df.columns):
        raise ValueError('중복된 Excel 열 이름이 있습니다.')
    student_col = "학생명" if "학생명" in df.columns else next(
        (col for col in ("학생", "이름", "student_id", '학번', 'ID', 'name', 'student_name') if col in df.columns), None
    )
    if student_col is None:
        raise ValueError("학생 식별 열(학생명/이름)이 없습니다.")
    qcols: dict[int, str] = {}
    for col in df.columns:
        if re.fullmatch(r'[Qq]0*\d+\.\d+', col):
            raise ValueError(f'중복된 문항 열이 있습니다: {col}')
        match = re.fullmatch(r"[Qq]0*(\d+)", col)
        if match:
            no = int(match.group(1))
            if no in qcols:
                raise ValueError(f"문항 열 {no}번이 중복입니다.")
            qcols[no] = col
    if set(qcols) != set(question_numbers):
        missing = sorted(set(question_numbers) - set(qcols))
        excess = sorted(set(qcols) - set(question_numbers))
        raise ValueError(f"OMR 문항과 정답 PDF 불일치: 누락={missing}, 초과={excess}")
    students: list[dict] = []
    names: set[str] = set()
    for index, row in df.iterrows():
        raw_name = row[student_col]
        if pd.isna(raw_name) or str(raw_name).strip() == "":
            # A completely empty row is ignored, but a row with answers is invalid.
            if all(pd.isna(row[qcols[n]]) for n in question_numbers):
                continue
            raise ValueError(f"Excel {index+2}행: 학생명이 없습니다.")
        name = str(raw_name).strip()
        if name in names:
            raise ValueError(f"중복 학생명: {name}")
        names.add(name)
        answers = {}
        for n in question_numbers:
            try:
                answers[n] = choice_number(row[qcols[n]])
            except ValueError as exc:
                raise ValueError(f"Excel {index+2}행 {name}, {n}번: {exc}") from exc
        students.append({"name": name, "answers": answers})
    if not students:
        raise ValueError("OMR에 학생 데이터가 없습니다.")
    if len(students) > 5000:
        raise ValueError("학생 수 제한(5,000명)을 초과했습니다.")
    return students
