import os
from dotenv import load_dotenv
from fastapi import FastAPI
from shared.database import build_mysql_url, make_engine, make_session_factory, Base
from .routes import build_router

load_dotenv()
app = FastAPI(title="AI Recommendation Engine", version="1.0.0")

db_url = build_mysql_url(
    host=os.environ["MYSQLHOST"],
    port=os.environ["MYSQLPORT"],
    user=os.environ["MYSQLUSER"],
    password=os.environ["MYSQLPASSWORD"],
    db=os.environ["MYSQLDATABASE"],
)
engine = make_engine(db_url)
SessionLocal = make_session_factory(engine)
Base.metadata.create_all(bind=engine)

app.include_router(build_router(SessionLocal), prefix="/ai", tags=["ai"])

@app.get("/health")
def health():
    return {"status": "healthy", "service": "ai-recommendation-engine"}
