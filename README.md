---
title: Customer Support OpenEnv
emoji: 🎧
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# Customer Support OpenEnv

> A real-world reinforcement learning environment where an AI agent learns to handle customer support tickets — classify issues, craft replies, escalate when needed, and resolve tickets.

Built for the **Meta × PyTorch OpenEnv Hackathon — Round 1, April 2026.**

---

## What is this?

Most RL environments are games. This one is not.

Every company with customers has a support queue. Tickets come in — billing complaints, app crashes, refund requests, angry users threatening legal action. A human agent reads each one, figures out what's wrong, replies helpfully, escalates if it's too serious, and closes it.

This environment teaches an AI to do exactly that. The agent receives a ticket, takes actions step by step, and gets rewarded based on how well it handles the situation. The reward signal is **dense** — the agent gets feedback at every step, not just at the end.

---

## Architecture

### Overall System

```mermaid
graph TD
    A[Agent] -->|Action| B[CustomerSupportEnv]
    B -->|Observation, Reward, Done, Info| A

    B --> C[tasks.py\nTask Definitions]
    B --> D[models.py\nTyped Models]
    B --> E[grader.py\nPer-Task Graders]

    F[app.py\nFastAPI Server] --> B
    G[baseline/run_baseline.py] --> B

    H[openenv.yaml\nMetadata + Config] -.->|describes| B
    I[data/tickets.json\nTicket Dataset] -.->|reference data| B
```

### Episode Flow

```mermaid
sequenceDiagram
    participant Agent
    participant Env as CustomerSupportEnv
    participant Grader

    Agent->>Env: reset(task_id="hard")
    Env-->>Agent: Observation (ticket + history + status)

    Agent->>Env: step(Action: classify, category="billing")
    Env-->>Agent: Observation, Reward(+0.3), done=False

    Agent->>Env: step(Action: reply, content="...")
    Env-->>Agent: Observation, Reward(+0.2), done=False

    Agent->>Env: step(Action: escalate)
    Env-->>Agent: Observation, Reward(+0.2), done=False

    Agent->>Env: step(Action: close)
    Env-->>Agent: Observation, Reward(+0.3), done=True

    Agent->>Grader: grade_task(task, actions_taken)
    Grader-->>Agent: Final Score (0.0 - 1.0)
```

### Reward Breakdown

```mermaid
flowchart LR
    A[Action Taken] --> B{action_type?}

    B -->|classify| C{Category correct?}
    C -->|yes| D[+0.30]
    C -->|no| E[+0.00]

    B -->|reply| F{Keyword hits?}
    F --> G[+0.10 per hit\nmax +0.40]
    F -->|replied before classify| H[-0.05 penalty]

    B -->|escalate| I{Required?}
    I -->|yes| J[+0.20]
    I -->|no| K[-0.10 penalty]

    B -->|close| L[+0.10 if classified\n+0.10 if replied\n+0.10 if escalated correctly]

    B -->|any, at max_steps| M[-0.05 time penalty]
```

### File Structure

```mermaid
graph LR
    root[customer-support-openenv]

    root --> env[env/]
    env --> models[models.py\nObservation Action Reward]
    env --> environment[environment.py\nCustomerSupportEnv]
    env --> tasks[tasks.py\nTask Definitions]
    env --> grader[grader.py\ngrade_easy grade_medium grade_hard]
    env --> utils[utils.py\nHelpers]
    env --> init[__init__.py]

    root --> baseline[baseline/]
    baseline --> script[run_baseline.py\nLLM + Mock runner]

    root --> data[data/]
    data --> tickets[tickets.json\n12 real tickets]

    root --> apppy[app.py\nFastAPI Server]
    root --> yaml[openenv.yaml]
    root --> docker[Dockerfile]
    root --> readme[README.md]
    root --> env2[.env\nAPI Keys]
```

---

## Tasks

The environment has 3 tasks of increasing difficulty. An agent must handle all three.

| Task | Difficulty | Max Steps | What the agent must do |
|---|---|---|---|
| `easy` | 🟢 Easy | 5 | Just classify the ticket correctly |
| `medium` | 🟡 Medium | 8 | Classify + give a helpful reply |
| `hard` | 🔴 Hard | 10 | Classify → reply → escalate → close |

### Easy — Classification Only
```
Customer: "I was charged twice for my order and need the duplicate removed."
Agent must → classify as "billing"
Score: 1.0 correct, 0.0 wrong
```

### Medium — Classify + Reply
```
Customer: "The app keeps crashing on my iPhone. I already restarted twice."
Agent must → classify as "technical" AND reply with relevant keywords
Score: 0.4 (classify) + up to 0.6 (reply quality)
```

### Hard — Full Pipeline
```
Customer: "Been waiting 3 weeks for my refund. Considering legal action."
History: 4 prior messages showing escalation attempts
Agent must → classify + reply + escalate to human + close ticket
Score: 0.2 + 0.3 + 0.2 + 0.3 (partial credit, penalty for bad escalation)
```

---

## Observation Space

What the agent sees at each step:

```python
Observation(
    ticket_id="T001",
    customer_query="I was charged twice and need a refund.",
    history=["Agent: We are looking into it.", "Customer: Still waiting!"],
    status="pending"   # open | pending | resolved
)
```

---

## Action Space

What the agent can do:

```python
Action(action_type="classify", category="billing")          # identify the issue
Action(action_type="reply",    content="We will help...")   # respond to customer
Action(action_type="escalate")                              # pass to human agent
Action(action_type="close")                                 # end the episode
```

Valid categories: `billing` | `technical` | `refund` | `account` | `abuse`

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd customer-support-openenv
pip install -r requirements.txt
```

### 2. Add your API key (optional — needed for LLM baseline)

```bash
# .env
OPENAI_API_KEY=sk-...
```

### 3. Run the baseline

```bash
python baseline/run_baseline.py
```

No API key? It runs in **mock mode** with deterministic actions — still produces valid scores.

### 4. Start the HTTP server

```bash
python app.py
# → http://localhost:7860
```

### 5. Try it manually

```bash
# Start a hard task episode
curl "http://localhost:7860/reset?task_id=hard"

# Classify the ticket
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "classify", "category": "billing"}'

# Reply
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "reply", "content": "We are escalating your refund as priority."}'

# Escalate
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "escalate"}'

# Close
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "close"}'
```

### 6. Use directly in Python

```python
from env import CustomerSupportEnv, Action

env = CustomerSupportEnv()
obs = env.reset(task_id="hard")

print(obs.customer_query)
# → "I have been waiting three weeks for a refund..."

obs, reward, done, info = env.step(Action(action_type="classify", category="billing"))
print(reward.score, reward.feedback)
# → 0.3  "correct category"

obs, reward, done, info = env.step(Action(
    action_type="reply",
    content="We are making this a priority refund and escalating to a manager."
))

obs, reward, done, info = env.step(Action(action_type="escalate"))
obs, reward, done, info = env.step(Action(action_type="close"))
```

---

## Docker

```bash
docker build -t openenv .
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... openenv
```

---

## Deploying to Hugging Face Spaces

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create a new Space → select **Docker** SDK
3. Add tag: `openenv`
4. Upload this entire repo
5. Add `OPENAI_API_KEY` as a Space secret

The server starts automatically and exposes all endpoints.

---

## Baseline Scores

Measured with deterministic mock actions (no API key needed):

| Task | Mock Score | LLM Score (gpt-4o-mini) |
|---|---|---|
| easy | 1.000 | ~0.900 |
| medium | 0.850 | ~0.750 |
| hard | 0.775 | ~0.650 |
| **Total** | **2.625 / 3.0** | **~2.300 / 3.0** |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | HTML landing page |
| GET | `/reset?task_id=easy` | Start a new episode |
| POST | `/step` | Submit an Action |
| GET | `/state` | Current raw state |
| GET | `/tasks` | List all tasks |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |

---

## Team

- **Adit Sharma** — adit.2428cs1345@kiet.edu
- **Mansi Verma** — ogmansi897@gmail.com
- **Priyanshi Vishwakarma** — vishwakarmapriyanshi68@gmail.com

---

*Meta × PyTorch OpenEnv Hackathon — Round 1, April 2026*
