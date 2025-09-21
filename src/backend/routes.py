from fastapi import APIRouter
from .supabase_client import supabase
from .models import Driver, Session, EndSession, DistractionEvent
from datetime import datetime

router = APIRouter()

@router.post("/drivers")
def create_driver(driver: Driver):
    res = supabase.table("drivers").insert(driver.dict()).execute()
    return {"status": "success", "data": res.data}

@router.post("/sessions/start")
def start_session(session: Session):
    res = supabase.table("sessions").insert({
        "driver_id": session.driver_id,
        "start_time": datetime.utcnow().isoformat()
    }).execute()
    return {"status": "success", "data": res.data}

@router.post("/sessions/end")
def end_session(session: EndSession):
    try:
        res = supabase.table("sessions").update({
            "end_time": datetime.utcnow().isoformat(),
            "avg_focus": session.avg_focus,
            "total_events": session.total_events
        }).eq("id", str(session.session_id)).execute()
        if res.error:
            return {"status": "error", "message": res.error.message}
        return {"status": "success", "data": res.data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/events")
def log_event(event: DistractionEvent):
    try:
        res = supabase.table("distraction_events").insert(event.dict()).execute()
        if res.error:
            return {"status": "error", "message": res.error.message}
        return {"status": "success", "data": res.data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

