import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
import asyncio
from paper_utils import search_papers, EventIndex, load_papers_from_url
from gemini_utils import cluster_papers, summarize_cluster, recluster_uncategorized_papers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="ACL Event Paper Search API")

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

# Initialize event index
current_index = None

class SearchRequest(BaseModel):
    query: str
    page: Optional[int] = 1
    per_page: Optional[int] = 10

class ClusterResponse(BaseModel):
    clusters: List[Dict[str, Any]]

class SummaryRequest(BaseModel):
    task_name: str
    paper_ids: List[str]

class EventRequest(BaseModel):
    event_url: str

class ReclusterRequest(BaseModel):
    paper_ids: List[str]
    iteration: int = 1

@app.post("/load_papers")
async def load_papers(request: EventRequest):
    """Load papers from an event URL."""
    try:
        papers = await load_papers_from_url(request.event_url)
        
        # Create event index which handles caching and embeddings
        global current_index
        current_index = EventIndex(papers, request.event_url)
        
        # Add embeddings to paper objects for the frontend
        for i, paper in enumerate(papers):
            paper['embedding'] = current_index.embeddings[i].tolist()
        
        return papers
        
    except Exception as e:
        logger.error(f"Error loading papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(request: SearchRequest):
    """Search papers using semantic search."""
    try:
        if not current_index:
            raise HTTPException(status_code=400, detail="No papers loaded. Please load papers first.")
        
        # Search with a higher limit to allow for pagination
        results = await search_papers(current_index, request.query, top_k=50)
        
        # Calculate pagination
        start_idx = (request.page - 1) * request.per_page
        end_idx = start_idx + request.per_page
        
        # Paginate the results
        paginated_results = results[start_idx:end_idx]
        
        return {
            "results": paginated_results,
            "total": len(results),
            "page": request.page,
            "per_page": request.per_page,
            "total_pages": (len(results) + request.per_page - 1) // request.per_page
        }
        
    except Exception as e:
        logger.error(f"Error searching papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cluster")
async def cluster_papers_endpoint(background_tasks: BackgroundTasks):
    """Start clustering papers."""
    try:
        if not current_index:
            raise HTTPException(status_code=400, detail="No papers loaded. Please load papers first.")
            
        # Get papers with embeddings from current index
        papers_with_embeddings = current_index.papers.copy()
        for i, paper in enumerate(papers_with_embeddings):
            paper['embedding'] = current_index.embeddings[i].tolist()
            
        # Start clustering in background
        background_tasks.add_task(cluster_papers_background, papers_with_embeddings)
        
        return {"status": "started"}
        
    except Exception as e:
        logger.error(f"Error starting clustering: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cluster_status")
async def get_cluster_status():
    """Get the current status of clustering and results if available."""
    try:
        if not hasattr(app.state, 'clusters'):
            return {"status": "clustering", "clusters": None}
        return {"status": "complete", "clusters": app.state.clusters}
    except Exception as e:
        logger.error(f"Error getting cluster status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def cluster_papers_background(papers_with_embeddings):
    """Background task for clustering papers."""
    try:
        logger.info("Starting background clustering task")
        if not papers_with_embeddings:
            logger.error("No papers loaded for clustering")
            app.state.clusters = {"clusters": []}
            return
            
        logger.info(f"Processing {len(papers_with_embeddings)} papers for clustering")
        
        logger.info(f"Starting clustering with {len(papers_with_embeddings)} papers")
        clusters = await cluster_papers(papers_with_embeddings)
        logger.info("Clustering completed, storing results")
        app.state.clusters = clusters.dict()
        logger.info("Clustering results stored in app state")
        
    except Exception as e:
        logger.error(f"Error in background clustering: {str(e)}", exc_info=True)
        app.state.clusters = {"clusters": []}

@app.post("/summarize")
async def get_cluster_summary(request: SummaryRequest) -> Dict[str, str]:
    """Get a summary for a specific cluster of papers."""
    if not current_index or not current_index.papers:
        raise HTTPException(status_code=400, detail="No papers loaded")
    
    try:
        # Get papers for the requested IDs
        cluster_papers = [p for p in current_index.papers if p['id'] in request.paper_ids]
        if not cluster_papers:
            raise HTTPException(status_code=404, detail="No papers found for the given IDs")
            
        summary = await summarize_cluster(request.task_name, cluster_papers, timeout=30)
        return {"summary": summary}
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Summarization operation timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recluster_uncategorized")
async def recluster_uncategorized(request: ReclusterRequest):
    """Recluster uncategorized papers."""
    try:
        if not current_index:
            raise HTTPException(status_code=400, detail="No papers loaded. Please load papers first.")
            
        # Get papers by IDs
        papers_to_cluster = []
        for paper in current_index.papers:
            if paper.get('id') in request.paper_ids:
                papers_to_cluster.append(paper)
                
        if not papers_to_cluster:
            raise HTTPException(status_code=400, detail="No papers found with provided IDs")
            
        # Recluster the papers
        clusters = await recluster_uncategorized_papers(papers_to_cluster, request.iteration)
        return {"clusters": clusters}
        
    except Exception as e:
        logger.error(f"Error reclustering papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    """Serve the main HTML page."""
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)