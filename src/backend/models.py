from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class Driver(BaseModel):
    name: str
    email: str
    license_number: Optional[str] = None
    role: Optional[str] = "driver"

class Session(BaseModel):
    driver_id: str

class EndSession(BaseModel):
    session_id: str
    avg_focus: float
    total_events: int

class DistractionEvent(BaseModel):
    session_id: str
    type: str
    start_time: datetime
    end_time: datetime
    duration_sec: Optional[float] = None
    confidence: Optional[float] = None
