from fastapi import APIRouter, Depends

from dgxmm.auth.deps import Principal, current_principal, get_ctx
from dgxmm.context import AppContext

router = APIRouter(prefix="/api/engines", tags=["engines"])


@router.get("")
def engines(p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    return [{"name": name, "supports_sleep": e.supports_sleep, "defaults": e.config_model().model_dump(mode="json"),
             "schema": e.config_model.model_json_schema(), "image": getattr(ctx.settings.images, name, None)}
            for name, e in ctx.engines.items()]
