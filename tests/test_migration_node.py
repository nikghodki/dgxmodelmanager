"""Migration 41: deployments.node column (per-node deployment tracking)."""
from dgxmm.migrations import migrate
from dgxmm.db import Database
from tests.nodes_fixtures import v40_db_with_one_deployment, row


def test_migration_41_backfills_local(v40_db_with_one_deployment):
    """Migration 41 adds node column and backfills 'local' for existing deployments."""
    # Migrate from v40 to latest
    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{v40_db_with_one_deployment}")
    migrate(engine)
    engine.dispose()
    
    # Check that node was backfilled to 'local'
    assert row(v40_db_with_one_deployment, "SELECT node FROM deployments") == ("local",)


def test_migration_41_default_is_a_string_literal(v40_db_with_one_deployment):
    """M13: DEFAULT "local" is an identifier in SQL (SQLite only falls back to a string); a literal is local."""
    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{v40_db_with_one_deployment}")
    migrate(engine)
    engine.dispose()
    (ddl,) = row(v40_db_with_one_deployment, "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'deployments'")
    assert "DEFAULT 'local'" in ddl and 'DEFAULT "local"' not in ddl
