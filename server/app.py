import os
import sys

# Ensure root directory is in path so we can import 'env'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from env.environment import CustomerSupportEnv
from env.models import Action
from env.tasks import TASKS
from fastapi import Request

app = FastAPI(title="Customer Support OpenEnv", version="1.0.0")

# one env per session
sessions = {}

def get_env(session_id="default"):
    if session_id not in sessions:
        sessions[session_id] = CustomerSupportEnv()
    return sessions[session_id]


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html><body style="font-family:sans-serif;background:#0f1117;color:#e0e0e0;max-width:700px;margin:50px auto;padding:0 24px">
    <h1 style="color:#7ee787">Customer Support OpenEnv</h1>
    <p>An OpenEnv RL environment for customer support automation.</p>
    <h2 style="color:#58a6ff">Endpoints</h2>
    <ul>
      <li><a href="/docs" style="color:#58a6ff">/docs</a> &mdash; Swagger UI</li>
      <li><code>GET /reset?task_id=easy|medium|hard</code></li>
      <li><code>POST /step</code> &mdash; send an Action</li>
      <li><code>GET /state</code></li>
      <li><a href="/tasks" style="color:#58a6ff">GET /tasks</a></li>
    </ul>
    </body></html>
    """


@app.get("/health")
def health():
    return {"status": "ok"}



@app.api_route("/reset", methods=["GET", "POST"])
async def reset(request: Request, task_id: str = None, session_id: str = "default"):
    if request.method == "POST":
        try:
            body = await request.json()
            task_id = body.get("task_id", task_id)
            session_id = body.get("session_id", session_id)
        except Exception:
            pass
            
    env = get_env(session_id)
    try:
        obs = env.reset(task_id=task_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {
        "observation": obs.model_dump(),
        "task": {
            "id": env.current_task["id"],
            "description": env.current_task["description"],
            "max_steps": env.current_task["max_steps"],
        },
    }


@app.post("/step")
def step(action: Action, session_id: str = "default"):
    env = get_env(session_id)
    if not env.current_task:
        raise HTTPException(400, "Call /reset first.")
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(session_id: str = "default"):
    env = get_env(session_id)
    if not env.current_task:
        raise HTTPException(400, "Call /reset first.")
    return env.state()


@app.get("/tasks")
def list_tasks():
    return [
        {
            "id": t["id"],
            "description": t["description"],
            "max_steps": t["max_steps"],
            "requires_escalation": t["expected"]["requires_escalation"],
        }
        for t in TASKS.values()
    ]


def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()