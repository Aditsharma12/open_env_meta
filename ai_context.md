# Customer Support OpenEnv Context

This document is designed to provide other AIs or LLMs with the full context of the project we are building. 
This project is for the **Meta × PyTorch OpenEnv Hackathon — Round 1**.

## Project Objective
We are building a real-world reinforcement learning environment called `CustomerSupportEnv`. 
Instead of a game, this environment simulates a customer support ticketing system. An AI agent must process tickets by taking distinct actions:
- **Classify**: Assign the correct category.
- **Reply**: Draft a response containing required keywords.
- **Escalate**: Pass it to human support if necessary.
- **Close**: End the episode.

The environment issues a **dense, shaped reward** at each step rather than a sparse end-of-episode signal.

## Core Files & Project Structure

The project has been refactored into a clean, modular structure.

*   `app.py`: FastAPI server used to interact with the environment over HTTP API, deploying to Hugging Face Spaces.
*   `env/models.py`: Pydantic schemas defining Observation, Action, and Reward.
*   `env/environment.py`: The actual RL environment logic maintaining state and calculating dense rewards.
*   `env/tasks.py`: Predefined tasks with expected goals (easy, medium, hard).
*   `openenv.yaml`: Metadata configuration describing the action space, observation space, tasks, and reward structure.

---

## Code Snippet: Pydantic State & Actions (`env/models.py`)

```python
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Observation(BaseModel):
    ticket_id: str
    customer_query: str
    history: List[str]
    status: str

class Action(BaseModel):
    action_type: str  # classify | reply | escalate | close
    content: Optional[str] = None
    category: Optional[str] = None

class Reward(BaseModel):
    score: float
    feedback: str
    breakdown: Dict[str, Any] = {}
```

## Code Snippet: Dense Reward Function (`env/environment.py`)

Here is how the environment issues incremental rewards based on the action taken:

```python
def _compute_reward(self, action: Action) -> Reward:
    correct = self.current_task["expected"]
    score = 0.0
    breakdown = {}

    if action.action_type == "classify":
        if action.category and action.category.lower() == correct["category"].lower():
            score += 0.3
            breakdown["classify"] = 0.3
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
        if self._classified: bonus += 0.1
        if self._replied: bonus += 0.1
        if correct["requires_escalation"] and self._escalated: bonus += 0.1
        score += bonus
        breakdown["close_bonus"] = bonus

    score = round(max(0.0, min(1.0, score)), 4)
    feedback = self._make_feedback(action, breakdown, correct)
    return Reward(score=score, feedback=feedback, breakdown=breakdown)
```

## Code Snippet: FastAPI Server & Workflow (`app.py`)

A minimal HTTP interface on port 7860 handles interaction:

```python
@app.get("/reset")
def reset(task_id: str = None, session_id: str = "default"):
    env = get_env(session_id)
    obs = env.reset(task_id=task_id)
    return {
        "observation": obs.model_dump(),
        "task": {
            "id": env.current_task["id"],
            "description": env.current_task["description"]
        }
    }

@app.post("/step")
def step(action: Action, session_id: str = "default"):
    env = get_env(session_id)
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }
```

## Code Snippet: Config details (`openenv.yaml`)

```yaml
# Observation Space
observation_space:
  type: structured
  fields:
    - name: ticket_id
    - name: customer_query
    - name: history
    - name: status

# Reward Structure
reward_structure:
  classify_correct: +0.3
  reply_per_keyword_hit: +0.1 (max 0.4)
  reply_before_classify: -0.05
  escalate_correct: +0.2
  escalate_unnecessary: -0.1
  close_bonus: +0.0 to +0.3 (depends on prior progress)
  time_penalty: -0.05 (if step_count >= max_steps)
```

## Usage Instructions for External LLM Systems

1. Familiarize yourself with the exact HTTP API and Payload requirements (`/reset`, `/step`).
2. Recognize that "Actions" must map accurately to `classify`, `reply`, `escalate`, or `close` and use corresponding kwargs (like `category` or `content`).
3. Ensure the LLM or agent understands it will receive incremental feedback. It should prioritize checking `reward.score` and `reward.breakdown` after every step to refine subsequent actions. 
4. Avoid "early replies" (replying before classifying) and penalizable acts like unnecessary unrequired escalations.
