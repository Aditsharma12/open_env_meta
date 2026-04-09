import sys
import os
import json
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from env.environment import CustomerSupportEnv
from env.models import Action
from env.grader import grade_task

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") # Optional for docker usage

from openai import OpenAI
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else os.getenv("OPENAI_API_KEY", "dummy_key"),
)

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

def call_llm(obs, messages):
    messages.append({"role": "user", "content": obs_to_text(obs)})
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        messages.append({"role": "assistant", "content": raw})
        return Action(**json.loads(raw))
    except Exception as e:
        return Action(action_type="close")

def run_llm(task_id):
    env = CustomerSupportEnv()
    obs = env.reset(task_id=task_id)
    task = env.current_task
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    taken = []

    print(f"[START] task={task_id}", flush=True)
    for i in range(task["max_steps"]):
        action = call_llm(obs, messages)
        obs, reward, done, info = env.step(action)
        taken.append(action)
        print(f"[STEP] step={i+1} reward={reward}", flush=True)
        if done:
            break

    score = grade_task(task, taken)
    print(f"[END] task={task_id} score={score} steps={len(taken)}", flush=True)
    return score

def main():
    for tid in ["easy", "medium", "hard"]:
        run_llm(tid)

if __name__ == "__main__":
    main()