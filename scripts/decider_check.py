"""Manual check: ask a running decider some golden-set items through dgxmm's SystemOneClient.

    python scripts/decider_check.py http://127.0.0.1:PORT KEYFILE GOLDEN.jsonl [N]

Prints id, gold, answer, confidence and latency per item (the key is read from KEYFILE and never printed)."""
import asyncio
import json
import sys
from pathlib import Path

import httpx

from dgxmm.decider.client import SystemOneClient
from dgxmm.decider.types import Option
from dgxmm.engines.decision import DECIDER_REVISION


async def main(base: str, keyfile: str, golden: str, n: int) -> None:
    key = Path(keyfile).read_text().strip()
    items = [json.loads(line) for line in open(golden)][:n]
    async with httpx.AsyncClient() as http:
        c = SystemOneClient(http, base_url=lambda: base, api_key=lambda: key, revision=DECIDER_REVISION)
        print("health:", await c.health(timeout_s=30))
        for it in items:
            opts = [Option(k, k, v or "") for k, v in it["options"].items()]
            a = await c.choose(feature="diagnosis", state=it["state"], question=it["question"], options=opts,
                               timeout_s=10)
            print(f"{it['id']} gold={it['gold']} answer={a.choice} p={a.confidence:.3f} "
                  f"{'OK' if a.choice == it['gold'] else 'MISS'} {a.latency_ms:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 3))
