"""Deterministic grading & statistics; NO LLM in scoring."""
from __future__ import annotations

from collections import Counter
from statistics import mean, median, pstdev


def grade_and_analyze(answer_key: dict[int, dict], students: list[dict]) -> dict:
    if not answer_key or not students:
        raise ValueError('정답표와 학생 답안이 모두 필요합니다.')
    if len({s['name'] for s in students}) != len(students):
        raise ValueError('중복 학생 식별자가 있습니다.')
    for n, item in answer_key.items():
        if type(n) is not int or type(item.get('answer')) is not int or not 1 <= item['answer'] <= 5 or type(item.get('points')) not in (int, float) or not 0 < item['points'] <= 10000:
            raise ValueError('정답표의 문항·정답·배점이 잘못되었습니다.')
    for student in students:
        if set(student['answers']) != set(answer_key):
            raise ValueError('학생 답안과 정답표의 문항이 다릅니다.')
        if any(v is not None and (type(v) is not int or not 1 <= v <= 5) for v in student['answers'].values()):
            raise ValueError('답안은 1~5 또는 미응답이어야 합니다.')
    nums = list(sorted(answer_key))
    max_score = sum(answer_key[n]["points"] for n in nums)
    if max_score <= 0:
        raise ValueError("총 배점이 0입니다.")
    reports = []
    misses = Counter({n: 0 for n in nums})
    for student in students:
        score, details = 0, {}
        for n in nums:
            correct = student["answers"][n] == answer_key[n]["answer"]
            # None (blank OMR answer) must NEVER be counted as correct.
            correct = bool(correct and student["answers"][n] is not None)
            details[n] = correct
            if correct:
                score += answer_key[n]["points"]
            else:
                misses[n] += 1
        reports.append({"name": student["name"], "score": score,
                        "answers": student["answers"], "correct": details})
    scores = [r["score"] for r in reports]
    worst = sorted(reports, key=lambda r: r["score"])
    lowest_score = worst[0]["score"]
    lowest_students = [r["name"] for r in worst if r["score"] == lowest_score]
    ranked_misses = sorted(nums, key=lambda n: (-misses[n], n))
    summary = {
        "student_count": len(reports), "question_count": len(nums),
        "max_score": max_score, "mean": mean(scores), "median": median(scores),
        "std_population": pstdev(scores),
        "lowest_score": lowest_score, "lowest_students": lowest_students,
        "most_missed": [{"number": n, "miss_count": misses[n],
                         "miss_rate": misses[n] / len(reports)} for n in ranked_misses[:3]],
    }
    question_stats = [{"number": n, "answer": answer_key[n]["answer"],
                       "points": answer_key[n]["points"], "miss_count": misses[n],
                       "correct_count": len(reports) - misses[n],
                       "miss_rate": misses[n] / len(reports)} for n in nums]
    return {"summary": summary, "students": reports,
            "question_stats": question_stats, "answer_key": answer_key}
