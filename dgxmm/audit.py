from dgxmm.db import AuditLog, Database


def audit(db: Database, user_id: int | None, action: str, target: str = "", detail: dict | None = None) -> None:
    with db.session() as s:
        s.add(AuditLog(user_id=user_id, action=action, target=target[:256], detail=detail or {}))
