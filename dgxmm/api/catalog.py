import re

from fastapi import APIRouter, Depends, Query

from dgxmm.api.errors import api_error
from dgxmm.auth.deps import Principal, current_principal, get_ctx
from dgxmm.context import AppContext
from dgxmm.inventory.artifact import REPO_RE

router = APIRouter(prefix="/api/catalog", tags=["catalog"])


@router.get("/search")
async def search(q: str = Query("", max_length=100), gguf: bool = False, ctx: int = Query(32768, ge=512, le=1_048_576),
                 p: Principal = Depends(current_principal), c: AppContext = Depends(get_ctx)):
    try:
        # viewers' lookups go out without the admin's HF token (no private or gated-access probing with it)
        return await c.catalog.search(q.strip(), gguf=gguf, ctx=ctx, anonymous=p.user.role != "admin")
    except Exception as e:
        raise api_error(502, "hub_error", f"Hugging Face search failed: {e}")


@router.get("/details")
async def details(repo: str, ctx: int = Query(32768, ge=512, le=1_048_576),
                  p: Principal = Depends(current_principal), c: AppContext = Depends(get_ctx)):
    if not re.fullmatch(REPO_RE, repo) or ".." in repo:
        raise api_error(422, "invalid_repo", "repo id must look like org/name")
    try:
        return await c.catalog.details(repo, ctx=ctx, anonymous=p.user.role != "admin")
    except Exception as e:
        raise api_error(502, "hub_error", f"could not load {repo}: {e}")
