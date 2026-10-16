from typing import Literal

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from dgxmm.api.auth import user_out
from dgxmm.api.errors import api_error
from dgxmm.audit import audit
from dgxmm.auth.deps import Principal, current_principal, get_ctx, require_admin
from dgxmm.auth.passwords import hash_password, validate_new_password
from dgxmm.auth.sessions import delete_user_sessions
from dgxmm.context import AppContext
from dgxmm.db import User

router = APIRouter(prefix="/api/users", tags=["users"])
USERNAME = r"^[a-z0-9_.-]{3,32}$"


class UserIn(BaseModel):
    username: str = Field(pattern=USERNAME)
    password: str
    role: Literal["admin", "viewer"]


class UserPatch(BaseModel):
    role: Literal["admin", "viewer"] | None = None
    password: str | None = None


@router.get("")
def list_users(p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    with ctx.db.session() as s:
        return [user_out(u) for u in s.scalars(select(User).order_by(User.username))]


@router.post("", status_code=201)
def create_user(body: UserIn, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    try:
        validate_new_password(body.password)
    except ValueError as e:
        raise api_error(422, "weak_password", str(e))
    with ctx.db.session() as s:
        if s.scalar(select(User).where(User.username == body.username)):
            raise api_error(409, "username_taken", "that username already exists")
        u = User(username=body.username, pw_hash=hash_password(body.password), role=body.role, must_change_pw=True)
        s.add(u)
        s.flush()
        out = user_out(u)
    audit(ctx.db, p.user.id, "user_created", body.username, {"role": body.role})
    return out


def _admin_count(s) -> int:
    return s.scalar(select(func.count()).select_from(User).where(User.role == "admin"))


@router.patch("/{user_id}")
def update_user(user_id: int, body: UserPatch, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    with ctx.db.session() as s:
        u = s.get(User, user_id)
        if u is None:
            raise api_error(404, "not_found", "no such user")
        if body.role and body.role != u.role:
            if u.role == "admin" and _admin_count(s) <= 1:
                raise api_error(409, "last_admin", "cannot demote the last admin")
            u.role = body.role
        if body.password:
            try:
                validate_new_password(body.password)
            except ValueError as e:
                raise api_error(422, "weak_password", str(e))
            u.pw_hash = hash_password(body.password)
            u.must_change_pw = True
        out = user_out(u)
    if body.password:
        delete_user_sessions(ctx.db, user_id)
    audit(ctx.db, p.user.id, "user_updated", out["username"], {"role": body.role, "password_reset": bool(body.password)})
    return out


@router.delete("/{user_id}", status_code=204)
def delete_user(user_id: int, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    if user_id == p.user.id:
        raise api_error(409, "cannot_delete_self", "you cannot delete your own account")
    with ctx.db.session() as s:
        u = s.get(User, user_id)
        if u is None:
            raise api_error(404, "not_found", "no such user")
        if u.role == "admin" and _admin_count(s) <= 1:
            raise api_error(409, "last_admin", "cannot delete the last admin")
        name = u.username
        s.delete(u)
    audit(ctx.db, p.user.id, "user_deleted", name)
    return Response(status_code=204)
