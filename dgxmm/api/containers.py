import asyncio

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from dgxmm.api.deployments import deploy_error
from dgxmm.audit import audit
from dgxmm.auth.deps import Principal, current_principal, get_ctx, require_admin
from dgxmm.context import AppContext
from dgxmm.deployer.adopt import adopt, list_adoptable
from dgxmm.deployer.service import DeployError

router = APIRouter(prefix="/api/containers", tags=["containers"])


class AdoptIn(BaseModel):
    name: str


@router.get("/adoptable")
async def adoptable(p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    return await asyncio.to_thread(list_adoptable, ctx.deployer)


@router.post("/{container}/adopt", status_code=201)
async def adopt_route(container: str, body: AdoptIn, p: Principal = Depends(require_admin),
                      ctx: AppContext = Depends(get_ctx)):
    try:
        view = await adopt(ctx.deployer, container, body.name, p.user.id)
    except DeployError as e:
        raise deploy_error(e)
    audit(ctx.db, p.user.id, "adopt", container, {"as": body.name})
    return view
