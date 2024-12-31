from typing import List, Dict, Any, Optional
from sklearn.preprocessing import normalize
import logging
import json
import re
import asyncio
from pydantic import BaseModel, Field
import numpy as np
import hdbscan
from dotenv import load_dotenv
import hashlib
import os

load_dotenv()

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize Gemini
import google.generativeai as genai

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

generation_config = genai.types.GenerationConfig(
    temperature=0.1,
    top_k=10,
)

class Cluster(BaseModel):
    task: str = Field(..., description="Name of the NLP task")
    paper_ids: List[str] = Field(..., description="List of paper IDs in this cluster")
    papers: List[Dict[str, Any]] = Field([], description="List of papers in this cluster")
    can_recluster: Optional[bool] = Field(False, description="Whether this cluster can be reclustered")

class ClusterResponse(BaseModel):
    clusters: List[Cluster] = Field(..., description="List of clusters")
    stats: Dict[str, Any] = Field({}, description="Clustering statistics")

class PaperSummary(BaseModel):
    summary: str = Field(..., description="Summary of the papers in the cluster")

GET_CLUSTER_TASK_PROMPT = """You are an expert in Natural Language Processing research.
Your task is to analyze a cluster of research papers and determine their common theme or task.

Here are the papers in this cluster:
{papers_json}

Instructions:
1. Analyze the titles and abstracts
2. Identify the common research theme or task
3. Return a concise but specific task name (3-7 words)
4. Focus on the main contribution or methodology
5. Be specific about the task or technique

Return ONLY the task name.
Do NOT include any explanatory text, markdown formatting, or JSON formatting in the response."""

SUMMARY_PROMPT = """You are an expert in Natural Language Processing research.
Create a comprehensive summary of the following papers that all investigate {task_name}.

Papers to summarize (in JSON format):
{papers_json}

Create a detailed summary that:
1. Identifies the main challenges and problems being addressed in this area
2. Highlights key methodological contributions and approaches
3. Synthesizes the main findings and results
4. Points out any emerging trends or common themes
5. Notes any significant contradictions or debates in the approaches

Format the summary in markdown with clear sections and bullet points.
Focus on extracting insights that would be valuable for researchers in this area."""

def _get_task_cache_key(papers: List[Dict[str, Any]]) -> str:
    """Generate a unique cache key for cluster task naming"""
    paper_ids = sorted([p.get('id', '') for p in papers])
    content_hash = hashlib.md5(''.join(paper_ids).encode()).hexdigest()[:16]
    return f"task_{content_hash}"

def _get_task_cache_path(cache_key: str) -> str:
    """Get cache file path for cluster task naming"""
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{cache_key}.txt")

def _load_task_cache(papers: List[Dict[str, Any]]) -> Optional[str]:
    """Load cached task name if it exists"""
    try:
        cache_key = _get_task_cache_key(papers)
        cache_path = _get_task_cache_path(cache_key)

        if os.path.exists(cache_path):
            logger.info(f"Found cached task for cluster: {cache_path}")
            with open(cache_path, 'r') as f:
                return f.read().strip()
    except Exception as e:
        logger.error(f"Error loading task cache: {e}")
    return None

def _save_task_cache(papers: List[Dict[str, Any]], task_name: str):
    """Save task name to cache"""
    try:
        cache_key = _get_task_cache_key(papers)
        cache_path = _get_task_cache_path(cache_key)

        logger.info(f"Saving task to cache: {cache_path}")
        with open(cache_path, 'w') as f:
            f.write(task_name)
    except Exception as e:
        logger.error(f"Error saving task cache: {e}")

async def get_cluster_task(papers: List[Dict[str, Any]]) -> str:
    """Get the specific NLP task for a cluster of papers using Gemini."""
    try:
        # Check cache first
        cached_task_name = _load_task_cache(papers)
        if cached_task_name:
            return cached_task_name

        # Prepare papers data
        papers_data = [{
            'title': paper['title']
        } for paper in papers]

        papers_json = json.dumps(papers_data, ensure_ascii=False)
        prompt = GET_CLUSTER_TASK_PROMPT.format(papers_json=papers_json)

        response = await model.generate_content_async(
            prompt,
            generation_config=generation_config,
        )

        # Get just the task name, removing any quotes or extra whitespace
        task_name = response.text.strip().strip('"\'')

        # Save to cache
        _save_task_cache(papers, task_name)

        return task_name

    except Exception as e:
        logger.error(f"Error getting cluster task: {str(e)}", exc_info=True)
        return "Miscellaneous"

async def recluster_uncategorized_papers(papers: List[Dict[str, Any]], iteration: int = 1) -> List[Dict[str, Any]]:
    """Recluster uncategorized papers using HDBSCAN and Gemini."""
    try:
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting reclustering of {len(papers)} papers (iteration {iteration})")
        logger.info(f"{'='*50}\n")

        # Get embeddings
        embeddings = [p.get('embedding', []) for p in papers]
        if not embeddings or not any(embeddings):
            raise ValueError("Papers missing embeddings")

        embeddings = np.array(embeddings, dtype=np.float32)
        embeddings = normalize(embeddings)

        # Apply HDBSCAN with more relaxed parameters for each iteration
        min_cluster_sizes = [10, 7, 5]
        min_samples = [3, 2, 1]

        if iteration > len(min_cluster_sizes):
            raise ValueError(f"Invalid iteration {iteration}. Maximum is {len(min_cluster_sizes)}")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_sizes[iteration-1],
            min_samples=min_samples[iteration-1],
            metric='euclidean',
            cluster_selection_method='eom'
        )

        cluster_labels = clusterer.fit_predict(embeddings)
        unique_labels = set(cluster_labels)
        n_noise = sum(1 for label in cluster_labels if label == -1)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        logger.info("\nHDBSCAN Results:")
        logger.info(f"  - Found {n_clusters} subclusters")
        logger.info(f"  - {n_noise} papers remain unclustered ({(n_noise/len(papers))*100:.1f}%)")

        # Group papers by cluster
        clusters_dict = {}
        uncategorized = []

        for idx, label in enumerate(cluster_labels):
            if label == -1:
                uncategorized.append(papers[idx])
            else:
                if label not in clusters_dict:
                    clusters_dict[label] = []
                clusters_dict[label].append(papers[idx])

        # Get names for all clusters at once
        tasks = []
        for cluster_papers in clusters_dict.values():
            tasks.append(get_cluster_task(cluster_papers))

        cluster_names = await asyncio.gather(*tasks)

        # Create final clusters list
        final_clusters = []
        for (label, cluster_papers), task_name in zip(clusters_dict.items(), cluster_names):
            # Ensure each paper has all necessary fields
            processed_papers = [{
                'id': p.get('id', ''),
                'title': p.get('title', ''),
                'authors': p.get('authors', []),
                'abstract': p.get('abstract', ''),
                'url': p.get('url', ''),
                'embedding': p.get('embedding', [])
            } for p in cluster_papers]

            final_clusters.append({
                "task": task_name,
                "papers": processed_papers,
                "paper_ids": [p['id'] for p in processed_papers]
            })

        # Add remaining uncategorized papers (only if they haven't been reclustered before)
        if uncategorized and iteration < len(min_cluster_sizes):
            # Process uncategorized papers too
            processed_uncategorized = [{
                'id': p.get('id', ''),
                'title': p.get('title', ''),
                'authors': p.get('authors', []),
                'abstract': p.get('abstract', ''),
                'url': p.get('url', ''),
                'embedding': p.get('embedding', [])
            } for p in uncategorized]

            final_clusters.append({
                "task": "Uncategorized",
                "papers": processed_uncategorized,
                "paper_ids": [p['id'] for p in processed_uncategorized],
                "can_recluster": True
            })

        logger.info("\nFinal Subclusters:")
        for cluster in final_clusters:
            if cluster["task"] != "Uncategorized":
                logger.info(f"  - {cluster['task']}: {len(cluster['papers'])} papers")

        return final_clusters

    except Exception as e:
        logger.error(f"Error in recluster_uncategorized_papers: {str(e)}", exc_info=True)
        raise

def _get_cache_key(papers: List[Dict[str, Any]]) -> str:
    """Generate a unique cache key based on paper contents"""
    # Sort paper IDs to ensure consistent cache key
    paper_ids = sorted([p.get('id', '') for p in papers])
    content_hash = hashlib.md5(''.join(paper_ids).encode()).hexdigest()[:16]
    return f"clusters_{content_hash}"

def _get_cache_path(cache_key: str) -> str:
    """Get cache file path for clusters"""
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{cache_key}.json")

def _load_clusters_cache(papers: List[Dict[str, Any]]) -> Optional[List[Cluster]]:
    """Load cached clusters if they exist"""
    try:
        cache_key = _get_cache_key(papers)
        cache_path = _get_cache_path(cache_key)

        if os.path.exists(cache_path):
            logger.info(f"Found cached clusters: {cache_path}")
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)

            # Convert cached data back to Cluster objects
            cached_clusters = []
            for cluster_data in cached_data:
                # Find papers for this cluster
                cluster_papers = []
                for paper in papers:
                    if paper.get('id') in cluster_data['paper_ids']:
                        cluster_papers.append(paper)

                cluster = Cluster(
                    task=cluster_data['task'],
                    paper_ids=cluster_data['paper_ids'],
                    papers=cluster_papers,
                    can_recluster=cluster_data.get('can_recluster', False)
                )
                cached_clusters.append(cluster)

            return cached_clusters
    except Exception as e:
        logger.error(f"Error loading clusters cache: {e}")
    return None

def _save_clusters_cache(papers: List[Dict[str, Any]], clusters: List[Cluster]):
    """Save clusters to cache"""
    try:
        cache_key = _get_cache_key(papers)
        cache_path = _get_cache_path(cache_key)

        # Convert clusters to serializable format
        serializable_clusters = []
        for cluster in clusters:
            serializable_cluster = {
                "task": cluster.task,
                "paper_ids": cluster.paper_ids,
                "can_recluster": cluster.can_recluster
            }
            serializable_clusters.append(serializable_cluster)

        logger.info(f"Saving clusters to cache: {cache_path}")
        with open(cache_path, 'w') as f:
            json.dump(serializable_clusters, f)
    except Exception as e:
        logger.error(f"Error saving clusters cache: {e}")

async def cluster_papers(papers: List[Dict[str, Any]]) -> ClusterResponse:
    """Cluster papers using HDBSCAN and Gemini."""
    try:
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting clustering of {len(papers)} papers")
        logger.info(f"{'='*50}\n")

        # Check cache first
        cached_clusters = _load_clusters_cache(papers)
        if cached_clusters:
            logger.info("Using cached clusters")
            return ClusterResponse(clusters=cached_clusters)

        # Get embeddings from papers and convert to numpy array
        logger.info("1. Extracting embeddings from papers...")
        embeddings = [p.get('embedding', []) for p in papers]
        if not embeddings or not any(embeddings):
            raise ValueError("Papers missing embeddings")

        # Convert list embeddings to numpy array
        embeddings = np.array(embeddings, dtype=np.float32)
        if len(embeddings.shape) != 2:
            raise ValueError(f"Invalid embedding shape: {embeddings.shape}. Expected 2D array.")

        # Normalize embeddings
        logger.info("2. Normalizing embeddings...")
        embeddings = normalize(embeddings)

        # Run HDBSCAN clustering
        logger.info("\n3. Running HDBSCAN clustering...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=15,
            min_samples=5,
            metric='euclidean',
            cluster_selection_method='eom'
        )

        cluster_labels = clusterer.fit_predict(embeddings)
        unique_labels = set(cluster_labels)
        n_noise = sum(1 for label in cluster_labels if label == -1)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        # Group papers by cluster
        clusters_dict = {}
        uncategorized_papers = []

        for idx, label in enumerate(cluster_labels):
            if label == -1:
                uncategorized_papers.append(papers[idx])
            else:
                if label not in clusters_dict:
                    clusters_dict[label] = []
                clusters_dict[label].append(papers[idx])

        # Get names for all clusters at once
        tasks = []
        logger.info(f"\n4. Sending {len(clusters_dict)} clusters to Gemini for naming...")
        for cluster_papers in clusters_dict.values():
            tasks.append(get_cluster_task(cluster_papers))

        cluster_names = await asyncio.gather(*tasks)

        # Create final clusters list
        final_clusters = []
        for (label, cluster_papers), task_name in zip(clusters_dict.items(), cluster_names):
            # Ensure each paper has all necessary fields
            processed_papers = [{
                'id': p.get('id', ''),
                'title': p.get('title', ''),
                'authors': p.get('authors', []),
                'abstract': p.get('abstract', ''),
                'url': p.get('url', ''),
                'embedding': p.get('embedding', [])
            } for p in cluster_papers]

            final_clusters.append(Cluster(
                task=task_name,
                paper_ids=[p['id'] for p in processed_papers],
                papers=processed_papers
            ))

        # Add uncategorized papers
        if uncategorized_papers:
            # Process uncategorized papers too
            processed_uncategorized = [{
                'id': p.get('id', ''),
                'title': p.get('title', ''),
                'authors': p.get('authors', []),
                'abstract': p.get('abstract', ''),
                'url': p.get('url', ''),
                'embedding': p.get('embedding', [])
            } for p in uncategorized_papers]

            final_clusters.append(Cluster(
                task="Uncategorized",
                paper_ids=[p['id'] for p in processed_uncategorized],
                papers=processed_uncategorized,
                can_recluster=len(processed_uncategorized) > 10  # Only show recluster button if enough papers
            ))

        # Log final clustering results
        logger.info("\nFinal Clustering Results:")
        logger.info(f"  - HDBSCAN identified: {n_clusters} clusters")
        logger.info(f"  - After Gemini naming: {len(final_clusters)} clusters")
        logger.info(f"  - Total papers: {len(papers)}")
        logger.info(f"  - Papers in clusters: {len(papers) - n_noise} ({((len(papers) - n_noise)/len(papers))*100:.1f}%)")
        logger.info(f"  - Unclustered papers: {n_noise} ({(n_noise/len(papers))*100:.1f}%)")

        # Save results to cache before returning
        _save_clusters_cache(papers, final_clusters)

        return ClusterResponse(clusters=final_clusters)

    except Exception as e:
        logger.error(f"Error in cluster_papers: {str(e)}", exc_info=True)
        # Return all papers in a single uncategorized cluster
        return ClusterResponse(clusters=[
            Cluster(
                task="Uncategorized",
                paper_ids=[p.get('id', '') for p in papers],
                papers=papers
            )
        ])

def _get_summary_cache_key(task_name: str, papers: List[Dict[str, Any]]) -> str:
    """Generate a unique cache key for cluster summaries"""
    paper_ids = sorted([p.get('id', '') for p in papers])
    content_hash = hashlib.md5((task_name + ''.join(paper_ids)).encode()).hexdigest()[:16]
    return f"summary_{content_hash}"

def _get_summary_cache_path(cache_key: str) -> str:
    """Get cache file path for cluster summaries"""
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{cache_key}.json")

def _load_summary_cache(task_name: str, papers: List[Dict[str, Any]]) -> Optional[str]:
    """Load cached summary if it exists"""
    try:
        cache_key = _get_summary_cache_key(task_name, papers)
        cache_path = _get_summary_cache_path(cache_key)

        if os.path.exists(cache_path):
            logger.info(f"Found cached summary for task '{task_name}': {cache_path}")
            with open(cache_path, 'r') as f:
                return json.load(f).get('summary')
    except Exception as e:
        logger.error(f"Error loading summary cache: {e}")
    return None

def _save_summary_cache(task_name: str, papers: List[Dict[str, Any]], summary: str):
    """Save summary to cache"""
    try:
        cache_key = _get_summary_cache_key(task_name, papers)
        cache_path = _get_summary_cache_path(cache_key)

        logger.info(f"Saving summary for task '{task_name}' to cache: {cache_path}")
        with open(cache_path, 'w') as f:
            json.dump({"summary": summary}, f)
    except Exception as e:
        logger.error(f"Error saving summary cache: {e}")

async def summarize_cluster(task_name: str, papers: List[Dict[str, Any]], timeout: int = 30, retry_count=3, initial_delay=1) -> PaperSummary:
    """
    Generate a summary for a cluster of papers with retry logic.
    """
    # Check for cached summary
    cached_summary = _load_summary_cache(task_name, papers)
    if cached_summary:
        return PaperSummary(summary=cached_summary)

    for attempt in range(retry_count):
        try:
            papers_json = json.dumps([{
                'title': paper['title'],
                'abstract': paper['abstract']
            } for paper in papers], ensure_ascii=False)

            prompt = SUMMARY_PROMPT.format(
                task_name=task_name,
                papers_json=papers_json
            )

            logger.info(f"Summarizing cluster: {task_name}, Number of papers: {len(papers)}")

            response = await asyncio.wait_for(model.generate_content_async(prompt), timeout=timeout)

            summary = response.text

            # Cache the summary
            _save_summary_cache(task_name, papers, summary)

            return PaperSummary(summary=summary)

        except asyncio.TimeoutError:
            logger.error(f"Summarization for {task_name} timed out (attempt {attempt+1}/{retry_count}).", exc_info=True)
            if attempt < retry_count - 1:
                delay = initial_delay * (2 ** attempt)
                logger.info(f"Retrying summarization for {task_name} in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"Max retries reached for summarization of {task_name}", exc_info=True)
                raise
        except Exception as e:
            logger.error(f"Error in summarizing cluster {task_name} (attempt {attempt+1}/{retry_count}): {str(e)}", exc_info=True)
            if attempt < retry_count -1:
                delay = initial_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"Max retries reached for batch, papers: {papers}", exc_info=True)
                raise
