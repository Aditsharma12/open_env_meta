import random
from copy import deepcopy
from .models import Observation, Action, Reward
from .tasks import TASKS, TASK_LIST


class CustomerSupportEnv:

    def __init__(self):
        self.current_task = None
        self.state_data = None
        self.done = False
        self.step_count = 0
        self._classified = False
        self._replied = False
        self._escalated = False
        self._closed = False

    def reset(self, task_id=None):
        if task_id:
            if task_id not in TASKS:
                raise ValueError(f"Unknown task '{task_id}'. Pick from: {list(TASKS.keys())}")
            self.current_task = TASKS[task_id]
        else:
            self.current_task = random.choice(TASK_LIST)

        self.state_data = deepcopy(self.current_task["input"])
        self.done = False
        self.step_count = 0
        self._classified = False
        self._replied = False
        self._escalated = False
        self._closed = False

        return Observation(**self.state_data)

    def step(self, action: Action):
        if self.done:
            raise RuntimeError("Episode done. Call reset() first.")

        self.step_count += 1
        reward = self._compute_reward(action)

        if action.action_type == "close":
            self.done = True
            self._closed = True

        # hit max steps → small penalty
        max_steps = self.current_task.get("max_steps", 10)
        if self.step_count >= max_steps and not self.done:
            self.done = True
            new_score = max(0.001, reward.score - 0.05)
            reward = Reward(
                score=new_score,
                feedback=reward.feedback + " | time limit hit, -0.05",
                breakdown={**reward.breakdown, "time_penalty": -0.05},
            )

        if action.content:
            self.state_data["history"].append(f"Agent: {action.content}")

        info = {
            "step": self.step_count,
            "task_id": self.current_task["id"],
            "classified": self._classified,
            "replied": self._replied,
            "escalated": self._escalated,
            "closed": self._closed,
        }

        return Observation(**self.state_data), reward, self.done, info

    def state(self):
        return self.state_data

    def _compute_reward(self, action: Action) -> Reward:
        correct = self.current_task["expected"]
        score = 0.0
        breakdown = {}

        if action.action_type == "classify":
            if action.category and action.category.lower() == correct["category"].lower():
                score += 0.3
                breakdown["classify"] = 0.3
            else:
                breakdown["classify"] = 0.0
            self._classified = True

        elif action.action_type == "reply":
            if not self._classified:
                score -= 0.05
                breakdown["early_reply_penalty"] = -0.05

            hits = sum(1 for kw in correct["keywords"] if kw in (action.content or "").lower())
            reply_score = min(0.4, hits * 0.1)
            score += reply_score
            breakdown["reply"] = reply_score
            self._replied = True

        elif action.action_type == "escalate":
            if correct["requires_escalation"]:
                score += 0.2
                breakdown["escalate"] = 0.2
            else:
                score -= 0.1
                breakdown["escalate"] = -0.1
            self._escalated = True

        elif action.action_type == "close":
            bonus = 0.0
            if self._classified:
                bonus += 0.1
            if self._replied:
                bonus += 0.1
            if correct["requires_escalation"] and self._escalated:
                bonus += 0.1
            score += bonus
            breakdown["close_bonus"] = bonus

        score = round(max(0.001, min(0.999, score)), 4)
        feedback = self._make_feedback(action, breakdown, correct)

        return Reward(score=score, feedback=feedback, breakdown=breakdown)

    def _make_feedback(self, action, breakdown, correct):
        parts = []

        if breakdown.get("classify") == 0.3:
            parts.append("correct category")
        elif "classify" in breakdown:
            parts.append(f"wrong category (expected {correct['category']})")

        if "early_reply_penalty" in breakdown:
            parts.append("replied before classifying")

        if "reply" in breakdown:
            parts.append(f"reply score {breakdown['reply']:.2f}")

        if breakdown.get("escalate") == 0.2:
            parts.append("escalated correctly")
        elif breakdown.get("escalate") == -0.1:
            parts.append("unnecessary escalation")

        if "close_bonus" in breakdown:
            parts.append(f"close bonus {breakdown['close_bonus']:.2f}")

        return ", ".join(parts) if parts else "ok"