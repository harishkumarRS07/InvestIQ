from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.core.config import settings
from backend.app.routes import router

from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from backend.data.update_stock_data import refined_update

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start scheduler
    scheduler = AsyncIOScheduler()
    # Schedule the update to run every day at 18:00 (6 PM)
    # You can adjust the time as needed. 18:00 is usually after market close.
    scheduler.add_job(refined_update, 'cron', hour=18, minute=0)
    scheduler.start()
    print("Scheduler started. Stock data update scheduled for 18:00 daily.")
    
    # Also run once on startup roughly to ensure data isn't stale? 
    # Maybe better not to block startup. 
    # Let's just rely on the schedule.
    
    yield
    
    # Shutdown scheduler
    scheduler.shutdown()

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    lifespan=lifespan
)

# CORS Security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {"message": "Welcome to Stock Predictor API. Use /api/v1/docs for documentation."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
