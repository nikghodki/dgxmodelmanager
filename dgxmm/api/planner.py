from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import select

from dgxmm.auth.deps import Principal, current_principal, get_ctx, require_admin
from dgxmm.context import AppContext
from dgxmm.db import Favourite
from dgxmm.planner.colocate import planner_view

router = APIRouter(prefix="/api/planner", tags=["planner"])


def _favs(ctx) -> set[str]:
    with ctx.db.session() as s:
        return {f.ref for f in s.scalars(select(Favourite))}


@router.get("")
async def planner(ctx_len: int | None = Query(None, alias="ctx", ge=512, le=1_048_576), must: list[str] = Query([]),
                  p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    budget = ctx.budget.current() or await ctx.budget.refresh()
    return planner_view(ctx.deployer, ctx.inventory, budget, _favs(ctx), ctx=ctx_len, must_include=tuple(must))


@router.put("/favourites/{ref:path}", status_code=204)
def add_fav(ref: str, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    with ctx.db.session() as s:
        if s.get(Favourite, ref) is None:
            s.add(Favourite(ref=ref[:512]))
    return Response(status_code=204)


@router.delete("/favourites/{ref:path}", status_code=204)
def del_fav(ref: str, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    with ctx.db.session() as s:
        f = s.get(Favourite, ref)
        if f:
            s.delete(f)
    return Response(status_code=204)
