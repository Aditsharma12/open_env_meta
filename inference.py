import sys
import os
import json
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from env.environment import CustomerSupportEnv
from env.models import Action
from env.grader import grade_task

SYSTEM_PROMPT = """You are an AI customer support agent inside an RL environment.
Read the ticket and respond with a JSON object ONLY. Pick one action:

{"action_type": "classify", "category": "<billing|technical|refund|account|abuse>"}
{"action_type": "reply", "content": "<your reply>"}
{"action_type": "escalate"}
{"action_type": "close"}

Strategy: classify first, reply next, escalate only if severe (legal threats / long-unresolved issues), then close."""


def obs_to_text(obs):
    lines = [f"Ticket: {obs.ticket_id}", f"Status: {obs.status}", f"Query: {obs.customer_query}"]
    if obs.history:
        lines.append("History:")
        for msg in obs.history:
            lines.append(f"  {msg}")
    return "\n".join(lines)


def call_llm(client, obs, messages):
    messages.append({"role": "user", "content": obs_to_text(obs)})
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        messages.append({"role": "assistant", "content": raw})
        return Action(**json.loads(raw))
    except Exception as e:
        print(f"  LLM error: {e}")
        return Action(action_type="close")


def run_llm(client, task_id):
    env = CustomerSupportEnv()
    obs = env.reset(task_id=task_id)
    task = env.current_task
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    taken = []

    print(f"\n{'='*55}")
    print(f"  Task: {task_id.upper()} | {task['description'][:50]}")
    print(f"{'='*55}")

    for i in range(task["max_steps"]):
        action = call_llm(client, obs, messages)
        obs, reward, done, info = env.step(action)
        taken.append(action)
        cat = f"cat={action.category}" if action.category else ""
        print(f"  step {i+1}: {action.action_type:<10} {cat:<16} reward={reward.score:.3f}")
        if done:
            break

    score = grade_task(task, taken)
    print(f"  grader score: {score:.3f}")
    return score


def run_mock(task_id):
    env = CustomerSupportEnv()
    env.reset(task_id=task_id)
    task = env.current_task
    ex = task["expected"]
    kw = ex["keywords"][0]

    actions = [
        Action(action_type="classify", category=ex["category"]),
        Action(action_type="reply", content=f"We understand your {ex['category']} issue. We will {kw} your request right away. Please reinstall if needed. Sorry for the inconvenience."),
    ]
    if ex["requires_escalation"]:
        actions.append(Action(action_type="escalate"))
    actions.append(Action(action_type="close"))

    taken = []
    print(f"\n{'='*55}")
    print(f"  Task: {task_id.upper()} | {task['description'][:50]}")
    print(f"{'='*55}")

    for action in actions:
        obs, reward, done, info = env.step(action)
        taken.append(action)
        cat = f"cat={action.category}" if action.category else ""
        print(f"  step {info['step']}: {action.action_type:<10} {cat:<16} reward={reward.score:.3f}")
        if done:
            break

    score = grade_task(task, taken)
    print(f"  grader score: {score:.3f}")
    return score


def main():
    api_key = os.getenv("OPENAI_API_KEY", "")
    use_llm = bool(api_key)

    print("\n[*] Customer Support OpenEnv - Baseline")
    print(f"    mode: {'LLM (gpt-4o-mini)' if use_llm else 'Mock (no API key)'}")

    client = None
    if use_llm:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

    results = {}
    for tid in ["easy", "medium", "hard"]:
        results[tid] = run_llm(client, tid) if use_llm else run_mock(tid)

    print(f"\n{'='*55}")
    print("  RESULTS")
    print(f"{'='*55}")
    for tid, score in results.items():
        bar = "#" * round(score * 25)
        print(f"  {tid:<10} {score:.3f}  {bar}")
    print(f"  {'total':<10} {sum(results.values()):.3f} / 3.000")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()