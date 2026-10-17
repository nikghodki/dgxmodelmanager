#!/usr/bin/env python3
"""Drive realistic traffic profiles against an OpenAI-compatible endpoint (one of YOUR OWN test deployments).

Evaluation tooling for the workload feature (docs/decider-evaluation.md); never run against user deployments.

    python -I scripts/decider_eval/traffic.py --base http://127.0.0.1:8102/v1 --model prod-h1-wl \
        --prompts ~/eval/prompts --docs ~/eval/docs --plan chat:18,agent:18,batch:18,rag:18,mixed:18,idle:18 \
        --log ~/eval/phases.jsonl

Profiles (each runs for its minutes, phases strictly sequential; the phase start/end times go to --log so the
metric rollups of each phase can be summarised separately):
- chat:  one person; multi-turn conversations, streamed, 8-45 s think time between turns.
- agent: two tool-calling agent loops; a long fixed system prompt + six tool schemas, real file contents returned
         as tool results, 0.3-2 s between steps, short outputs.
- batch: 24 workers sending questions back to back (more than max_num_seqs, so requests queue).
- rag:   one or two clients; ~6k-12k-token prompts assembled from random document chunks + a question, short
         answers, 2-10 s between queries.
- mixed: a chat user, a 60 s burst of 12 batch workers every ~3.5 min, an agent loop and a RAG query now and then.
- idle:  no requests.
Prompts: GSM8K / MMLU questions (local copies). Documents / tool results: real text files.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx

FOLLOW_UPS = ["Can you explain that more simply?", "Why does that work?", "Give me a concrete example.",
              "What would change if the numbers were doubled?", "Summarise your answer in two sentences.",
              "Is there a faster way to get the same result?", "What is a common mistake people make here?",
              "Can you check your answer step by step?", "How would you explain this to a ten-year-old?",
              "Thanks. One more question: how confident are you?"]
CHAT_OPENERS = ["I'm planning a three-day trip to Pune in December. What should I pack and see?",
                "Write a short, friendly email asking my landlord to fix a leaking tap.",
                "What's the difference between a process and a thread? Keep it practical.",
                "Help me plan a 30-minute beginner workout I can do at home without equipment.",
                "Explain how a heat pump works.", "Suggest five names for a small bakery that sells sourdough.",
                "What are good questions to ask at the end of a job interview?",
                "Can you give me a simple recipe for dal tadka?",
                "How do I politely decline a meeting invitation?",
                "Explain recursion with an everyday analogy."]
AGENT_SYSTEM = (
    "You are a careful software-engineering agent working inside a repository. You can inspect files, search "
    "the code, run tests and edit files ONLY through the provided tools. Work step by step: first explore, then "
    "form a plan, then act. Always call exactly one tool per step unless you are completely done; when you are "
    "done, reply with a short summary of what you found or changed and stop calling tools. Never invent file "
    "contents; read them. Prefer small, verifiable steps. If a command fails, read the error and adapt. Keep "
    "answers concise; the user reads only your final summary.\n\n" + "Repository conventions:\n" + "\n".join(
        f"- Rule {i}: " + r for i, r in enumerate([
            "Python code lives under dgxmm/, tests under tests/, the React UI under frontend/src.",
            "Every behaviour change needs a test that fails first.", "Never print secrets or tokens.",
            "Use the existing API helpers (api_error, require_admin) for new endpoints.",
            "Database changes go through db.add_missing_columns.", "Keep functions small and typed.",
            "Run the whole test suite before you finish.", "Prefer reading a file over guessing its contents.",
            "Do not touch deployment containers you did not create.", "Document user-visible changes."] * 4)))
TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file of the repository.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "file path"}},
                    "required": ["path"]}}},
    {"type": "function", "function": {"name": "search_code", "description": "Search the repository for a regex.",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "glob": {"type": "string"}},
                    "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "list_dir", "description": "List a directory.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run pytest on a path; returns output.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace text in a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"},
                                                     "new": {"type": "string"}}, "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
]
AGENT_TASKS = ["Find where deployment failures are diagnosed and summarise the rules used.",
               "Which function chooses a free port for a new deployment? Read it and explain it.",
               "Look for the code that redacts secrets from logs and list what it masks.",
               "Find the tests for the advisor rules and tell me which rule has the fewest tests.",
               "Check how the inventory decides the kind of a model and summarise it.",
               "Find the LiteLLM sync code and explain when a model is registered."]


class Stats:
    def __init__(self):
        self.ok = self.err = self.tool_calls = 0
        self.prompt_tokens = self.completion_tokens = 0

    def add(self, usage: dict | None, ok: bool = True, tool_calls: int = 0):
        if ok:
            self.ok += 1
        else:
            self.err += 1
        self.tool_calls += tool_calls
        if usage:
            self.prompt_tokens += usage.get("prompt_tokens") or 0
            self.completion_tokens += usage.get("completion_tokens") or 0

    def as_dict(self):
        return dict(vars(self))


class Driver:
    def __init__(self, base: str, model: str, prompts: Path, docs: Path, seed: int):
        self.base, self.model = base.rstrip("/"), model
        self.rng = random.Random(seed)
        self.gsm = json.loads((prompts / "gsm8k.json").read_text())
        self.mmlu = json.loads((prompts / "mmlu.json").read_text())
        self.docs = [p for p in sorted(docs.rglob("*")) if p.is_file() and p.suffix in (".md", ".py", ".txt")]
        self.chunks = []
        for p in self.docs:
            t = p.read_text(errors="replace")
            for i in range(0, len(t), 3000):
                self.chunks.append((p.name, t[i:i + 3000]))
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
        self.stats = Stats()

    def question(self) -> str:
        if self.rng.random() < 0.5:
            return self.rng.choice(self.gsm)
        m = self.rng.choice(self.mmlu)
        opts = "\n".join(f"{'ABCD'[i]}. {c}" for i, c in enumerate(m["choices"]))
        return f"{m['q']}\n{opts}\nAnswer with the letter and a one-line reason."

    async def complete(self, messages, *, max_tokens=256, tools=None, stream=False, temperature=0.7):
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        if tools:
            body["tools"], body["tool_choice"] = tools, "auto"
        try:
            if stream:
                body["stream"], body["stream_options"] = True, {"include_usage": True}
                text, usage = [], None
                async with self.http.stream("POST", f"{self.base}/chat/completions", json=body) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line.startswith("data: ") or line.endswith("[DONE]"):
                            continue
                        ev = json.loads(line[6:])
                        usage = ev.get("usage") or usage
                        for ch in ev.get("choices") or []:
                            text.append((ch.get("delta") or {}).get("content") or "")
                self.stats.add(usage)
                return {"role": "assistant", "content": "".join(text)}
            r = await self.http.post(f"{self.base}/chat/completions", json=body)
            r.raise_for_status()
            out = r.json()
            msg = out["choices"][0]["message"]
            self.stats.add(out.get("usage"), tool_calls=len(msg.get("tool_calls") or []))
            return msg
        except (httpx.HTTPError, ValueError, KeyError) as e:
            self.stats.add(None, ok=False)
            print(f"  request failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
            await asyncio.sleep(2)
            return None

    # ---- profiles ----
    async def chat_user(self, until: float, think=(8, 45)):
        while time.time() < until:
            opener = self.rng.choice(CHAT_OPENERS) if self.rng.random() < 0.4 else self.question()
            msgs = [{"role": "user", "content": opener}]
            for _ in range(self.rng.randint(3, 8)):
                if time.time() >= until:
                    return
                reply = await self.complete(msgs, max_tokens=self.rng.randint(150, 450), stream=True)
                if reply is None:
                    break
                msgs.append({"role": "assistant", "content": reply.get("content") or ""})
                await asyncio.sleep(min(self.rng.uniform(*think), max(0.0, until - time.time())))
                msgs.append({"role": "user", "content": self.rng.choice(FOLLOW_UPS)})

    def tool_result(self, name: str, args: dict) -> str:
        if name == "read_file" or name == "run_tests":
            p = self.rng.choice(self.docs)
            t = p.read_text(errors="replace")
            start = self.rng.randrange(0, max(1, len(t) - 2400))
            return f"# {p.name}\n" + t[start:start + 2400]
        if name == "search_code":
            name_, chunk = self.rng.choice(self.chunks)
            lines = [ln for ln in chunk.splitlines() if ln.strip()][:12]
            return "\n".join(f"{name_}:{i + 1}: {ln}" for i, ln in enumerate(lines))
        if name == "list_dir":
            return "\n".join(sorted({p.name for p in self.rng.sample(self.docs, min(25, len(self.docs)))}))
        if name == "get_weather":
            return json.dumps({"city": args.get("city", "?"), "temp_c": self.rng.randint(12, 34), "sky": "clear"})
        return "ok: 1 replacement"

    async def agent_loop(self, until: float, gap=(0.3, 2.0)):
        while time.time() < until:
            msgs = [{"role": "system", "content": AGENT_SYSTEM},
                    {"role": "user", "content": self.rng.choice(AGENT_TASKS)}]
            for _ in range(self.rng.randint(4, 10)):
                if time.time() >= until:
                    return
                msg = await self.complete(msgs, max_tokens=256, tools=TOOLS, temperature=0.3)
                if msg is None:
                    break
                calls = msg.get("tool_calls") or []
                msgs.append({"role": "assistant", "content": msg.get("content") or "",
                             **({"tool_calls": calls} if calls else {})})
                if not calls:
                    msgs.append({"role": "user", "content": "Continue: use the tools to check one more file."})
                for c in calls:
                    try:
                        args = json.loads(c["function"].get("arguments") or "{}")
                    except ValueError:
                        args = {}
                    msgs.append({"role": "tool", "tool_call_id": c.get("id", "call"),
                                 "content": self.tool_result(c["function"]["name"], args)})
                await asyncio.sleep(self.rng.uniform(*gap))

    async def batch_worker(self, until: float):
        while time.time() < until:
            await self.complete([{"role": "user", "content": self.question()}], max_tokens=256)

    def rag_prompt(self) -> str:
        target = self.rng.randint(7_000, 13_000) * 3  # characters: ~6k-12k tokens, inside a 16k context
        parts, n = [], 0
        while n < target:
            name, chunk = self.rng.choice(self.chunks)
            parts.append(f"[Document: {name}]\n{chunk}")
            n += len(chunk)
        q = self.rng.choice(["Which of these documents mentions configuration options, and what are they?",
                             "Summarise the main purpose of the first document in three sentences.",
                             "List three function or class names that appear in the documents.",
                             "Does any document talk about errors or failures? Quote one line."])
        return "Answer using only the documents below.\n\n" + "\n\n".join(parts) + f"\n\nQuestion: {q}"

    async def rag_client(self, until: float, gap=(2, 10)):
        while time.time() < until:
            await self.complete([{"role": "user", "content": self.rag_prompt()}], max_tokens=150, temperature=0.2)
            await asyncio.sleep(min(self.rng.uniform(*gap), max(0.0, until - time.time())))

    async def mixed(self, until: float):
        async def bursts():
            while time.time() < until:
                await asyncio.sleep(min(self.rng.uniform(150, 260), max(0.0, until - time.time())))
                end = min(until, time.time() + 60)
                await asyncio.gather(*(self.batch_worker(end) for _ in range(12)))

        async def occasional(fn, every):
            while time.time() < until:
                await asyncio.sleep(min(self.rng.uniform(*every), max(0.0, until - time.time())))
                if time.time() < until:
                    await fn(min(until, time.time() + self.rng.uniform(30, 90)))

        await asyncio.gather(self.chat_user(until), bursts(), occasional(self.agent_loop, (120, 300)),
                             occasional(self.rag_client, (180, 360)))

    async def run_phase(self, name: str, minutes: float):
        until = time.time() + minutes * 60
        if name == "chat":
            await self.chat_user(until)
        elif name == "agent":
            await asyncio.gather(self.agent_loop(until), self.agent_loop(until))
        elif name == "batch":
            await asyncio.gather(*(self.batch_worker(until) for _ in range(24)))
        elif name == "rag":
            await asyncio.gather(self.rag_client(until), self.rag_client(until, gap=(5, 15)))
        elif name == "mixed":
            await self.mixed(until)
        elif name == "idle":
            await asyncio.sleep(max(0.0, until - time.time()))
        else:
            raise SystemExit(f"unknown profile {name}")
        await asyncio.sleep(max(0.0, until - time.time()))


async def amain(args) -> int:
    d = Driver(args.base, args.model, Path(args.prompts), Path(args.docs), args.seed)
    for item in args.plan.split(","):
        name, minutes = item.split(":")
        d.stats = Stats()
        t0 = time.time()
        print(f"== {name} for {minutes} min", flush=True)
        await d.run_phase(name, float(minutes))
        rec = {"phase": name, "start": t0, "end": time.time(), **d.stats.as_dict()}
        print(json.dumps(rec), flush=True)
        with open(args.log, "a") as f:
            f.write(json.dumps(rec) + "\n")
    await d.http.aclose()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--seed", type=int, default=7)
    return asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
