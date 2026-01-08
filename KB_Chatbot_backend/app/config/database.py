from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base

DATABASE_URL = "sqlite+aiosqlite:///data/chat_history.db"

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

Base = declarative_base()
