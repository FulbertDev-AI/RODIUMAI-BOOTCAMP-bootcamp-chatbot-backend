"""PostgreSQL dialect checks.

Unit tests keep an in-memory SQLite override (see conftest.py) so the HTTP/SSE
suite stays fast and does not need a running server. This module checks that
SQLAlchemy models compile to PostgreSQL DDL. Live Alembic + FastAPI against a
real cluster is documented in README.md (TEST_DATABASE_URL / local Postgres).
"""

import os

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from database.models import Conversation, Message

LIVE_PG = os.getenv("TEST_DATABASE_URL", "")


def test_models_compile_to_postgresql_ddl():
    for table in (Conversation.__table__, Message.__table__):
        sql = str(CreateTable(table).compile(dialect=postgresql.dialect()))
        assert "CREATE TABLE" in sql.upper()
        assert "mysql" not in sql.lower()
        assert "pymysql" not in sql.lower()
    messages_sql = str(CreateTable(Message.__table__).compile(dialect=postgresql.dialect()))
    assert "conversation_id" in messages_sql
    assert "seq" in messages_sql
    conv_sql = str(CreateTable(Conversation.__table__).compile(dialect=postgresql.dialect()))
    assert "created_at" in conv_sql


def test_unique_and_fk_names_are_explicit():
    names = {c.name for c in Message.__table__.constraints}
    assert "uq_messages_conversation_id_seq" in names
    fk_names = {fk.name for fk in Message.__table__.foreign_keys}
    assert "fk_messages_conversation_id" in fk_names


@pytest.mark.skipif(
    not LIVE_PG.startswith("postgresql"),
    reason="Set TEST_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:PORT/DATABASE to run live PostgreSQL checks",
)
def test_live_postgres_connect_and_schema():
    from sqlalchemy import create_engine, inspect, text

    engine = create_engine(LIVE_PG, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    assert {"conversations", "messages"}.issubset(tables)
    msg_cols = {c["name"] for c in inspector.get_columns("messages")}
    assert {"id", "role", "content", "created_at", "conversation_id", "seq"} <= msg_cols
    unique = inspector.get_unique_constraints("messages")
    assert any(u["name"] == "uq_messages_conversation_id_seq" for u in unique)
    fks = inspector.get_foreign_keys("messages")
    assert any(fk["name"] == "fk_messages_conversation_id" for fk in fks)
    engine.dispose()
