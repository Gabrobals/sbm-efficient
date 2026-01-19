"""
VERTEX-RESEARCH API Server

FastAPI server che riceve webhook da n8n e integra con il tracker locale.

Endpoints:
    POST /project/new - Crea nuovo progetto
    POST /hypothesis/register - Registra ipotesi
    POST /experiment/start - Avvia esperimento
    POST /experiment/complete - Completa esperimento
    POST /daily - Log giornaliero
    GET /status - Stato sistema
    GET /projects - Lista progetti
"""

import os
import json
import httpx
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import del tracker locale
from src.utils.vertex_simple import VertexSimple, Config

app = FastAPI(
    title="VERTEX-RESEARCH API",
    description="API per integrare n8n con il VERTEX-RESEARCH Protocol",
    version="1.0.0"
)

# CORS per permettere chiamate da n8n
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tracker instance
tracker = VertexSimple()

# n8n webhook URL per notifiche inverse
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook")


# ==================== MODELS ====================

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = ""

class HypothesisCreate(BaseModel):
    project_id: Optional[str] = None
    statement: str
    conditions: str  # comma-separated
    falsification: str  # comma-separated
    thresholds: Optional[Dict[str, float]] = None

class ExperimentStart(BaseModel):
    project_id: Optional[str] = None
    config: Dict[str, Any]
    
class ExperimentComplete(BaseModel):
    experiment_id: str
    results: Dict[str, Any]

class ExperimentCheckpoint(BaseModel):
    experiment_id: str
    progress: float
    metrics: Dict[str, float]
    notes: Optional[str] = ""

class DailyLog(BaseModel):
    project_id: Optional[str] = None
    notes: str
    blockers: Optional[str] = ""
    hours: Optional[float] = 0

class HypothesisEvaluate(BaseModel):
    hypothesis_id: str
    results: Dict[str, float]

class GateReview(BaseModel):
    project_id: Optional[str] = None
    phase: int
    checklist_results: Dict[str, bool]


# ==================== HELPER ====================

async def notify_n8n(event: str, data: Dict[str, Any]):
    """Invia notifica a n8n webhook."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{N8N_WEBHOOK_URL}/vertex-event",
                json={"event": event, "data": data, "timestamp": datetime.now().isoformat()}
            )
    except Exception as e:
        print(f"n8n notification failed: {e}")


# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "ok",
        "service": "VERTEX-RESEARCH API",
        "version": "1.0.0",
        "n8n_webhook": N8N_WEBHOOK_URL
    }


@app.get("/status")
async def get_status():
    """Stato del sistema."""
    projects = tracker.storage.list_all(Config.PROJECTS_DIR)
    experiments = tracker.storage.list_all(Config.EXPERIMENTS_DIR)
    hypotheses = tracker.storage.list_all(Config.HYPOTHESES_DIR)
    
    return {
        "current_project": tracker.current_project_id,
        "counts": {
            "projects": len(projects),
            "experiments": len(experiments),
            "hypotheses": len(hypotheses)
        },
        "storage_path": str(Config.VERTEX_DIR)
    }


# ==================== PROJECTS ====================

@app.post("/project/new")
async def create_project(data: ProjectCreate, background_tasks: BackgroundTasks):
    """Crea nuovo progetto."""
    try:
        project = tracker.project_new(data.name)
        
        # Notifica n8n in background
        background_tasks.add_task(
            notify_n8n, 
            "project_created", 
            {"id": project.id, "name": project.name, "phase": project.phase}
        )
        
        return {
            "success": True,
            "project": {
                "id": project.id,
                "name": project.name,
                "phase": project.phase,
                "phase_name": project.phase_name,
                "created": project.created
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/projects")
async def list_projects():
    """Lista tutti i progetti."""
    from src.utils.vertex_simple import Project
    projects = tracker.storage.list_all(Config.PROJECTS_DIR, Project)
    
    return {
        "current": tracker.current_project_id,
        "projects": [
            {
                "id": p.id,
                "name": p.name,
                "phase": p.phase,
                "phase_name": p.phase_name,
                "status": p.status,
                "created": p.created
            }
            for p in projects
        ]
    }


@app.post("/project/{project_id}/advance")
async def advance_project(project_id: str, background_tasks: BackgroundTasks):
    """Avanza progetto alla prossima fase."""
    try:
        project = tracker.project_advance(project_id)
        
        background_tasks.add_task(
            notify_n8n,
            "phase_advanced",
            {"id": project.id, "phase": project.phase, "phase_name": project.phase_name}
        )
        
        return {
            "success": True,
            "project": {
                "id": project.id,
                "phase": project.phase,
                "phase_name": project.phase_name
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== HYPOTHESES ====================

@app.post("/hypothesis/register")
async def register_hypothesis(data: HypothesisCreate, background_tasks: BackgroundTasks):
    """Registra nuova ipotesi."""
    try:
        hypothesis = tracker.hypothesis_register(
            statement=data.statement,
            conditions=data.conditions,
            falsification=data.falsification,
            thresholds=data.thresholds,
            project_id=data.project_id
        )
        
        background_tasks.add_task(
            notify_n8n,
            "hypothesis_registered",
            {
                "id": hypothesis.id,
                "project_id": hypothesis.project_id,
                "statement": hypothesis.statement
            }
        )
        
        return {
            "success": True,
            "hypothesis": {
                "id": hypothesis.id,
                "statement": hypothesis.statement,
                "status": hypothesis.status,
                "registered": hypothesis.registered
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/hypothesis/evaluate")
async def evaluate_hypothesis(data: HypothesisEvaluate, background_tasks: BackgroundTasks):
    """Valuta ipotesi contro risultati."""
    try:
        evaluation = tracker.hypothesis_evaluate(data.hypothesis_id, data.results)
        
        background_tasks.add_task(
            notify_n8n,
            "hypothesis_evaluated",
            {
                "id": data.hypothesis_id,
                "status": evaluation["status"],
                "checks": evaluation["checks"]
            }
        )
        
        return {
            "success": True,
            "evaluation": evaluation
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/hypotheses")
async def list_hypotheses(project_id: Optional[str] = None):
    """Lista ipotesi."""
    from src.utils.vertex_simple import Hypothesis
    hypotheses = tracker.storage.list_all(Config.HYPOTHESES_DIR, Hypothesis)
    
    if project_id:
        hypotheses = [h for h in hypotheses if h.project_id == project_id]
    
    return {
        "hypotheses": [
            {
                "id": h.id,
                "project_id": h.project_id,
                "statement": h.statement,
                "status": h.status,
                "registered": h.registered
            }
            for h in hypotheses
        ]
    }


# ==================== EXPERIMENTS ====================

@app.post("/experiment/start")
async def start_experiment(data: ExperimentStart, background_tasks: BackgroundTasks):
    """Avvia nuovo esperimento."""
    try:
        experiment = tracker.experiment_start(data.config, data.project_id)
        
        background_tasks.add_task(
            notify_n8n,
            "experiment_started",
            {
                "id": experiment.id,
                "project_id": experiment.project_id,
                "config": experiment.config,
                "git_commit": experiment.git_commit
            }
        )
        
        return {
            "success": True,
            "experiment": {
                "id": experiment.id,
                "config": experiment.config,
                "started": experiment.started,
                "git_commit": experiment.git_commit
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/experiment/checkpoint")
async def checkpoint_experiment(data: ExperimentCheckpoint, background_tasks: BackgroundTasks):
    """Log checkpoint esperimento."""
    try:
        tracker.experiment_checkpoint(
            data.experiment_id,
            data.progress,
            data.metrics,
            data.notes
        )
        
        background_tasks.add_task(
            notify_n8n,
            "experiment_checkpoint",
            {
                "id": data.experiment_id,
                "progress": data.progress,
                "metrics": data.metrics
            }
        )
        
        return {"success": True, "progress": data.progress}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/experiment/complete")
async def complete_experiment(data: ExperimentComplete, background_tasks: BackgroundTasks):
    """Completa esperimento."""
    try:
        experiment = tracker.experiment_complete(data.experiment_id, data.results)
        
        background_tasks.add_task(
            notify_n8n,
            "experiment_completed",
            {
                "id": experiment.id,
                "results": experiment.results,
                "duration": experiment.completed
            }
        )
        
        return {
            "success": True,
            "experiment": {
                "id": experiment.id,
                "results": experiment.results,
                "completed": experiment.completed
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/experiments")
async def list_experiments(project_id: Optional[str] = None, status: Optional[str] = None):
    """Lista esperimenti."""
    from src.utils.vertex_simple import Experiment
    experiments = tracker.storage.list_all(Config.EXPERIMENTS_DIR, Experiment)
    
    if project_id:
        experiments = [e for e in experiments if e.project_id == project_id]
    if status:
        experiments = [e for e in experiments if e.status == status]
    
    return {
        "experiments": [
            {
                "id": e.id,
                "project_id": e.project_id,
                "config": e.config,
                "status": e.status,
                "started": e.started,
                "completed": e.completed,
                "results": e.results
            }
            for e in experiments
        ]
    }


# ==================== DAILY LOG ====================

@app.post("/daily")
async def create_daily_log(data: DailyLog, background_tasks: BackgroundTasks):
    """Crea log giornaliero."""
    try:
        log = tracker.daily_log(
            notes=data.notes,
            blockers=data.blockers,
            hours=data.hours,
            project_id=data.project_id
        )
        
        background_tasks.add_task(
            notify_n8n,
            "daily_logged",
            {
                "date": log.date,
                "project_id": log.project_id,
                "has_blockers": bool(log.blockers)
            }
        )
        
        return {
            "success": True,
            "log": {
                "date": log.date,
                "project_id": log.project_id,
                "notes": log.notes
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== REPORTS ====================

@app.get("/report/weekly")
async def generate_weekly_report():
    """Genera report settimanale."""
    report = tracker.report_weekly()
    return {"success": True, "report": report}


@app.get("/report/standup")
async def generate_standup():
    """Genera daily standup."""
    standup = tracker.report_daily_standup()
    return {"success": True, "standup": standup}


# ==================== GATE REVIEW ====================

@app.post("/gate/review")
async def submit_gate_review(data: GateReview, background_tasks: BackgroundTasks):
    """Sottomette risultati gate review (chiamato da n8n dopo form compilato)."""
    project_id = data.project_id or tracker.current_project_id
    
    all_passed = all(data.checklist_results.values())
    
    result = {
        "project_id": project_id,
        "phase": data.phase,
        "passed": all_passed,
        "checklist": data.checklist_results,
        "reviewed_at": datetime.now().isoformat()
    }
    
    # Salva risultato gate review
    gate_path = Config.VERTEX_DIR / "gate_reviews"
    gate_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"gate_{project_id}_phase{data.phase}_{datetime.now().strftime('%Y%m%d')}.json"
    with open(gate_path / filename, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Se passato, avanza automaticamente
    if all_passed:
        tracker.project_advance(project_id)
        
        background_tasks.add_task(
            notify_n8n,
            "gate_passed",
            {"project_id": project_id, "phase": data.phase}
        )
    else:
        background_tasks.add_task(
            notify_n8n,
            "gate_failed",
            {
                "project_id": project_id,
                "phase": data.phase,
                "failed_items": [k for k, v in data.checklist_results.items() if not v]
            }
        )
    
    return {
        "success": True,
        "passed": all_passed,
        "result": result
    }


# ==================== WEBHOOK RECEIVER ====================

@app.post("/webhook/n8n")
async def receive_n8n_webhook(payload: Dict[str, Any]):
    """
    Riceve webhook da n8n per azioni automatiche.
    
    n8n può chiamare questo endpoint per triggerare azioni
    basate su eventi esterni (es. commit GitHub, timer, etc.)
    """
    action = payload.get("action")
    data = payload.get("data", {})
    
    if action == "daily_reminder":
        # n8n ci ricorda di fare il daily log
        return {"message": "Daily reminder received", "pending": True}
    
    elif action == "experiment_timeout":
        # Esperimento in timeout
        exp_id = data.get("experiment_id")
        if exp_id:
            # Marca come failed
            from src.utils.vertex_simple import Experiment
            path = Config.EXPERIMENTS_DIR / f"{exp_id}.json"
            if path.exists():
                exp = tracker.storage.load(path, Experiment)
                exp.status = "timeout"
                exp.completed = datetime.now().isoformat()
                tracker.storage.save(exp, path)
                return {"message": f"Experiment {exp_id} marked as timeout"}
    
    elif action == "weekly_report":
        report = tracker.report_weekly()
        return {"message": "Weekly report generated", "report": report}
    
    return {"message": "Unknown action", "action": action}


# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
