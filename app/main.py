from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import asyncio
import uuid
from app.models import ResearchTask
from app.agents import ResearchCrew
from app.db import graph_db
from typing import Dict
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

app = FastAPI(title="Danish Medical Platform API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
tasks: Dict[str, ResearchTask] = {}

async def run_crew_task(task_id: str, topic: str):
    logger.info(f"Starting task {task_id} for topic: {topic}")
    tasks[task_id].status = "running"
    tasks[task_id].logs.append("Starting research crew...")
    
    try:
        crew = ResearchCrew(task_id)
        # This is blocking, but we are in a background thread (FastAPI BackgroundTasks run in threadpool)
        # Wait, FastAPI BackgroundTasks run in the same loop if async, or threadpool if def.
        # ResearchCrew.run is sync (CrewAI default).
        result = await asyncio.to_thread(crew.run, topic)
        
        tasks[task_id].status = "completed"
        tasks[task_id].result = {"output": str(result)}
        tasks[task_id].logs.append("Research completed successfully.")
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        tasks[task_id].status = "failed"
        tasks[task_id].logs.append(f"Error: {str(e)}")

@app.post("/research", response_model=ResearchTask)
async def start_research(request: dict, background_tasks: BackgroundTasks):
    topic = request.get("topic")
    if not topic:
        raise HTTPException(status_code=400, detail="Topic required")
        
    task_id = str(uuid.uuid4())
    task = ResearchTask(task_id=task_id, topic=topic)
    tasks[task_id] = task
    
    background_tasks.add_task(run_crew_task, task_id, topic)
    
    return task

@app.get("/status/{task_id}")
async def stream_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator():
        last_log_idx = 0
        while True:
            task = tasks.get(task_id)
            if not task:
                break
                
            # Check for new logs
            if len(task.logs) > last_log_idx:
                for log in task.logs[last_log_idx:]:
                    yield {"event": "log", "data": log}
                last_log_idx = len(task.logs)
            
            # Check status
            if task.status in ["completed", "failed"]:
                yield {"event": "status", "data": task.status}
                if task.result:
                    yield {"event": "result", "data": str(task.result)}
                break
                
            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())

@app.get("/report/{task_id}")
def get_report(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

@app.get("/graph")
def get_graph():
    return graph_db.get_graph_data()

@app.get("/")
def root():
    return {"message": "Danish Medical Platform API is running"}

