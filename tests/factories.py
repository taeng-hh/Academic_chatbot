"""Fresh binary inputs, independent of the supplied sample exam."""
import io
import pandas as pd
import pymupdf


def key_pdf(rows, order=('number', 'answer', 'points'), borders=True):
    labels = {'number': 'Question', 'answer': 'Answer', 'points': 'Points'}
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=500, height=650)
        matrix = [[labels[k] for k in order]] + [[str(row[k]) for k in order] for row in rows]
        for ri, row in enumerate(matrix):
            for ci, value in enumerate(row):
                page.insert_text((45 + ci*120, 58 + ri*30), value, fontsize=11)
        if borders:
            for ri in range(len(matrix)+1):
                page.draw_line((35, 35 + ri*30), (395, 35 + ri*30))
            for ci in range(4):
                page.draw_line((35 + ci*120, 35), (35 + ci*120, 35 + len(matrix)*30))
        return pdf.tobytes()


def omr_xlsx(students, id_column='student_id'):
    rows = [{id_column: s['name'], **{f'Q{n:02}': a for n, a in s['answers'].items()}} for s in students]
    output = io.BytesIO()
    pd.DataFrame(rows).to_excel(output, index=False)
    return output.getvalue()


def problem_pdf(numbers):
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=600, height=800)
        for i, n in enumerate(numbers):
            page.insert_text((40, 50 + i*140), f'{n}. Choose the best title for this independent passage.', fontsize=10)
            page.insert_text((40, 70 + i*140), 'Forests provide habitats and help protect the climate.', fontsize=10)
            page.insert_text((40, 90 + i*140), 'A) Trees B) Water C) Cities D) Roads E) Boats', fontsize=10)
        return pdf.tobytes()
