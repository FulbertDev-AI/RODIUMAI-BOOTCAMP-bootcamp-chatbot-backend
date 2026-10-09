import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

# Format : postgresql+psycopg://USER:PASSWORD@HOST:PORT/DATABASE
DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(engine)

class Base(DeclarativeBase):
    pass

def get_db():
    with SessionLocal() as db:
        yield db