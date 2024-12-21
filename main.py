from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
from paper_utils import get_paper_info, search_papers, EventIndex

app = FastAPI(title="ACL Paper Search API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class SearchQuery(BaseModel):
    query: str
    event_url: str  # Added to specify which event to search in
    top_k: Optional[int] = 10

class Paper(BaseModel):
    title: str
    authors: List[str]
    abstract: Optional[str]
    link: str

# Dictionary to store active event indexes
event_indexes: Dict[str, EventIndex] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize cache directory on startup"""
    os.makedirs("cache", exist_ok=True)

@app.get("/")
async def root():
    """Serve the main HTML page"""
    return FileResponse("static/index.html")

@app.get("/refresh/{event_url:path}")
async def refresh_papers(event_url: str):
    """Fetch papers from a given ACL event URL and update embeddings"""
    # Ensure the URL is from aclanthology.org
    if not event_url.startswith("https://aclanthology.org/"):
        raise HTTPException(status_code=400, detail="Invalid URL. Must be from aclanthology.org")
    
    # Fetch or load event index
    event_index = get_paper_info(event_url)
    if not event_index:
        raise HTTPException(status_code=500, detail="Failed to fetch papers")
    
    # Store the event index
    event_indexes[event_url] = event_index
    
    return {
        "message": f"Successfully fetched {len(event_index.papers)} papers",
        "paper_count": len(event_index.papers)
    }

@app.post("/search")
async def search(query: SearchQuery):
    """Search for papers using semantic search within a specific event"""
    event_index = event_indexes.get(query.event_url)
    if not event_index:
        raise HTTPException(
            status_code=400,
            detail="Event not loaded. Please fetch papers first using /refresh endpoint"
        )
    
    results = search_papers(query.query, event_index, top_k=query.top_k)
    return results

@app.get("/papers/{event_url:path}")
async def get_papers(event_url: str, skip: int = 0, limit: int = 10):
    """Get a list of papers with pagination for a specific event"""
    event_index = event_indexes.get(event_url)
    if not event_index:
        raise HTTPException(
            status_code=400,
            detail="Event not loaded. Please fetch papers first using /refresh endpoint"
        )
    
    return event_index.papers[skip:skip + limit]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
