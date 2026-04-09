from typing import List
from .models import Action


def grade_easy(task, actions: List[Action]) -> float:
    expected = task["expected"]["category"].lower()
    for a in actions:
        if a.action_type == "classify":
            return 0.999 if (a.category or "").lower() == expected else 0.001
    return 0.001


def grade_medium(task, actions: List[Action]) -> float:
    score = 0.0
    expected_cat = task["expected"]["category"].lower()
    keywords = [k.lower() for k in task["expected"]["keywords"]]

    for a in actions:
        if a.action_type == "classify":
            if (a.category or "").lower() == expected_cat:
                score += 0.4
            break

    for a in actions:
        if a.action_type == "reply" and a.content:
            hits = sum(1 for k in keywords if k in a.content.lower())
            score += min(0.6, hits * 0.15)
            break

    return round(max(0.001, min(0.999, score)), 4)


def grade_hard(task, actions: List[Action]) -> float:
    score = 0.0
    expected_cat = task["expected"]["category"].lower()
    keywords = [k.lower() for k in task["expected"]["keywords"]]
    needs_escalation = task["expected"]["requires_escalation"]

    for a in actions:
        if a.action_type == "classify":
            if (a.category or "").lower() == expected_cat:
                score += 0.2
            break

    for a in actions:
        if a.action_type == "reply" and a.content:
            hits = sum(1 for k in keywords if k in a.content.lower())
            score += min(0.3, hits * 0.075)
            break

    escalated = any(a.action_type == "escalate" for a in actions)
    if needs_escalation and escalated:
        score += 0.2
    elif not needs_escalation and escalated:
        score -= 0.1

    if any(a.action_type == "close" for a in actions):
        score += 0.3

    return round(max(0.001, min(0.999, score)), 4)


GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


def grade_task(task, actions: List[Action]) -> float:
    grader = GRADERS.get(task.get("id", "easy"), grade_easy)
    return grader(task, actions)