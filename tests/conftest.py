import os

# Set before any app import so load_dotenv does not require a real key or live database.
os.environ["RODIUMAI_API_KEY"] = "test-key"
os.environ["DATABASE_URL"] = "sqlite://"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from database.db import Base, get_db
from main import app


@pytest.fixture
def sqlite_sessionmaker():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield sessionmaker(engine)


@pytest.fixture
def client(sqlite_sessionmaker):
    def override_get_db():
        with sqlite_sessionmaker() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
