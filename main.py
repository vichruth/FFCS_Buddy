# File: main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
import engine

app = FastAPI(
    title="FFCS Buddy API v2",
    description="API for multi-filter faculty search and recommendation."
)

# Update the request model to include the new filters
class UserPreferences(BaseModel):
    course_code: Optional[str] = Field(None, example="CSE1004")
    slot: Optional[str] = Field(None, example="A1")
    query: Optional[str] = Field(None, example="an engaging teacher")
    style_prefs: Optional[List[str]] = Field(None, example=["project-based"])
    top_n: int = 10

@app.post("/find-courses")
def find_courses_endpoint(preferences: UserPreferences):
    """
    Handles multi-filter search for courses and faculty.
    """
    recommendations_df = engine.find_faculty(
        course_code=preferences.course_code,
        slot=preferences.slot,
        query=preferences.query,
        style_prefs=preferences.style_prefs,
        top_n=preferences.top_n
    )
    
    return {
        "status": "success",
        "results": recommendations_df.to_dict('records')
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to the FFCS Buddy Multi-Filter API!"}