from typing import List, Dict, Any, Optional
import logging
import json
import re
import asyncio
from pydantic import BaseModel, Field
import numpy as np
import hdbscan
from dotenv import load_dotenv
load_dotenv()
from sklearn.preprocessing import normalize
from paper_utils import get_text_for_embedding, model

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize Gemini
import google.generativeai as genai
import os

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

class ClusterResponse(BaseModel):
    clusters: List[Cluster] = Field(..., description="List of clusters")
    stats: Dict[str, Any] = Field({}, description="Clustering statistics")

class PaperSummary(BaseModel):
    summary: str = Field(..., description="Summary of the papers in the cluster")

CLUSTERING_PROMPT = '''You are an expert in Natural Language Processing research.
Your task is to analyze and cluster research papers based on their titles into broad NLP research areas.

Papers to analyze:
{papers_json}

Instructions:
1. Group these papers into coherent research areas or tasks
2. Each cluster should have a descriptive name that reflects the common theme
3. Return ONLY a JSON object with the following structure, with NO trailing commas:
{{
  "clusters": [
    {{
      "task": "Name of Research Area",
      "paper_ids": ["id1", "id2"]
    }}
  ]
}}

IMPORTANT: 
- Return ONLY the JSON object, no other text or formatting
- Do NOT use trailing commas in JSON arrays or objects
- Make sure the JSON is complete and valid'''

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

TASK_NAMING_PROMPT = '''You are an expert in Natural Language Processing research.
Your task is to analyze the titles of papers in a cluster and determine the specific NLP task or research area they focus on.

Papers in this cluster:
{papers_json}

Instructions:
1. Analyze the paper titles to identify the common NLP task or research focus
2. Be specific - avoid generic terms like "LLMs" or "Text Generation"
3. Focus on the concrete task (e.g., "Multilingual Named Entity Recognition" instead of just "NLP")
4. If papers seem too diverse, focus on the dominant theme

Return ONLY a single line with the specific task name, no additional text or explanation.'''

LARGE_CLUSTER_PROMPT = """Given a list of academic papers, create focused clusters based on their research areas and topics.
Requirements:
- Create 5-8 meaningful clusters that represent distinct research areas
- Each cluster should have a clear, specific focus
- Cluster names should be concise but descriptive (e.g. "Machine Translation", "Question Answering")
- Avoid overly broad or vague cluster names
- Ensure each paper is assigned to the most relevant cluster
- Papers that don't fit well should go to "Uncategorized"

Papers to cluster (in JSON format):
{papers_json}

Return ONLY a JSON object with this exact structure:
{{
    "clusters": [
        {{
            "task": "cluster name",
            "paper_ids": ["id1", "id2", ...]
        }},
        ...
    ]
}}
"""

MERGE_CLUSTERS_PROMPT = '''You are an expert in Natural Language Processing research.
Your task is to analyze and merge similar clusters of research papers into 10-15 meaningful categories.

Current clusters with their task names and paper counts:
{clusters_info}

Instructions:
1. Merge similar clusters to create 10-15 final categories
2. Each final category should have a clear and specific theme
3. Only create categories that have at least 15 papers
4. Try to create balanced categories where possible
5. Keep truly unique clusters separate
6. Put small or unclear clusters into "Uncategorized"

IMPORTANT: Return ONLY a JSON object with this structure (no additional text):
{{
  "merged_clusters": [
    {{
      "task": "Final Task Name",
      "source_clusters": ["Source Task 1", "Source Task 2"]
    }}
  ]
}}

Do not include any explanatory text, markdown formatting, or code blocks. Return ONLY the JSON object.
'''

async def extract_json_from_response(response_text: str) -> Dict:
    """Extract JSON from a response that might contain markdown."""
    try:
        # First try direct JSON parsing
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
            
        # Try to extract JSON from markdown code block
        if '```json' in response_text:
            # Extract content between ```json and ```
            pattern = r'```json\s*(.*?)\s*```'
            matches = re.findall(pattern, response_text, re.DOTALL)
            if matches:
                json_str = matches[0]
                # Clean up common JSON issues
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)  # Remove trailing commas
                json_str = re.sub(r',\s*$', '', json_str)  # Remove trailing comma at end of string
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse extracted JSON: {e}")
                    
                    # Try to salvage partial JSON
                    try:
                        # Find complete cluster objects
                        clusters = []
                        cluster_pattern = r'\{\s*"task":\s*"([^"]+)",\s*"paper_ids":\s*\[((?:"[^"]+",?\s*)*)\]\s*\}'
                        for match in re.finditer(cluster_pattern, json_str):
                            task = match.group(1)
                            paper_ids = re.findall(r'"([^"]+)"', match.group(2))
                            if task and paper_ids:
                                clusters.append({
                                    "task": task,
                                    "paper_ids": paper_ids
                                })
                        if clusters:
                            return {"clusters": clusters}
                    except Exception as e:
                        logger.error(f"Failed to salvage partial JSON: {e}")
                    raise
                    
        raise ValueError("No valid JSON found in response")
        
    except Exception as e:
        logger.error(f"Error extracting JSON from response: {str(e)}")
        logger.error(f"Response text: {response_text[:500]}...")
        raise

async def process_batch(batch: List[Dict[str, Any]], prompt_template: str, retry_count=3, initial_delay=1) -> Dict:
    """Process a single batch of papers using Gemini API with retry logic and rate limiting"""
    last_error = None
    for attempt in range(retry_count):
        try:
            papers_json = json.dumps(batch, ensure_ascii=False, indent=2)
            prompt = prompt_template.format(papers_json=papers_json)
            
            logger.info(f"Sending prompt (length: {len(prompt)}), batch size: {len(batch)}")
            
            response = await model.generate_content_async(
                prompt,
                generation_config=generation_config,
            )
            
            response_text = response.text
            logger.debug(f"Raw response: {response_text}")
            
            if response_text.strip().startswith('"clusters"') or response_text.strip().startswith('"Clusters"'):
                response_text = '{' + response_text + '}'
                
            if '"clusters":' in response_text and not response_text.strip().startswith('{'):
                response_text = '{' + response_text.split('"clusters":', 1)[1]
                response_text = '{"clusters":' + response_text
                
            logger.debug(f"Processed response: {response_text}")
            
            clusters_dict = await extract_json_from_response(response_text)
            return clusters_dict
        
        except Exception as e:
            last_error = e
            logger.error(f"Error processing batch (attempt {attempt+1}/{retry_count}): {str(e)}, Batch: {batch}", exc_info=True)
            logger.error(f"Prompt sent: {prompt}")
            
            if attempt < retry_count - 1:
                delay = initial_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"Max retries reached for batch, Batch: {batch}", exc_info=True)
                raise last_error

async def get_cluster_task(papers: List[Dict[str, Any]]) -> str:
    """Get the specific NLP task for a cluster of papers using Gemini."""
    try:
        # Prepare papers data
        papers_data = [{
            'title': paper['title']
        } for paper in papers]
        
        papers_json = json.dumps(papers_data, ensure_ascii=False)
        prompt = TASK_NAMING_PROMPT.format(papers_json=papers_json)
        
        response = await model.generate_content_async(
            prompt,
            generation_config=generation_config,
        )
        
        # Get just the task name, removing any quotes or extra whitespace
        task_name = response.text.strip().strip('"\'')
        return task_name
        
    except Exception as e:
        logger.error(f"Error getting cluster task: {str(e)}", exc_info=True)
        return "Miscellaneous NLP Tasks"

async def recluster_large_group(papers: List[Dict[str, Any]], max_papers_per_batch: int = 50) -> List[Cluster]:
    """Use Gemini to recluster a large group of papers."""
    try:
        logger.info(f"  - Preparing {len(papers)} papers for Gemini...")
        
        # Prepare paper data with titles and abstracts
        papers_data = [{
            'id': paper.get('id', ''),
            'title': paper.get('title', ''),
            'abstract': paper.get('abstract', '')[:300]  # Further truncate abstract to keep prompt size manageable
        } for paper in papers]
        
        # Process in smaller batches to avoid timeouts
        all_clusters = []
        batch_count = (len(papers_data) + max_papers_per_batch - 1) // max_papers_per_batch
        if batch_count > 1:
            logger.info(f"    - Processing in {batch_count} batches of {max_papers_per_batch} papers")
        
        for i in range(0, len(papers_data), max_papers_per_batch):
            batch = papers_data[i:i + max_papers_per_batch]
            batch_num = i // max_papers_per_batch + 1
            
            try:
                # Get clusters from Gemini
                papers_json = json.dumps(batch, ensure_ascii=False)
                prompt = LARGE_CLUSTER_PROMPT.format(papers_json=papers_json)
                
                if batch_count > 1:
                    logger.info(f"    - Sending batch {batch_num}/{batch_count} to Gemini...")
                
                response = await model.generate_content_async(
                    prompt,
                    generation_config=generation_config
                )
                
                if not response or not response.text:
                    logger.error(f"      - Empty response from Gemini for batch {batch_num}")
                    continue
                
                try:
                    clusters_data = json.loads(response.text)
                    batch_clusters = [
                        Cluster(
                            task=c['task'],
                            paper_ids=c['paper_ids'],
                            papers=[]  # Will be filled later
                        )
                        for c in clusters_data['clusters']
                    ]
                    if batch_count > 1:
                        logger.info(f"      - Batch {batch_num} created {len(batch_clusters)} clusters")
                    all_clusters.extend(batch_clusters)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from batch {batch_num}: {e}\nResponse: {response.text[:500]}")
                    continue
                    
            except Exception as batch_error:
                logger.error(f"Error processing batch {batch_num}: {str(batch_error)}", exc_info=True)
                continue
        
        if not all_clusters:
            logger.warning("    - No valid clusters created, returning uncategorized cluster")
            return [Cluster(
                task="Uncategorized",
                paper_ids=[p.get('id', '') for p in papers],
                papers=papers
            )]
        
        return all_clusters
        
    except Exception as e:
        logger.error(f"Error in recluster_large_group: {str(e)}", exc_info=True)
        return [Cluster(
            task="Uncategorized",
            paper_ids=[p.get('id', '') for p in papers],
            papers=papers
        )]

async def cluster_papers(papers: List[Dict[str, Any]]) -> ClusterResponse:
    """Cluster papers using HDBSCAN and Gemini."""
    try:
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting clustering of {len(papers)} papers")
        logger.info(f"{'='*50}\n")
        
        # Get embeddings from papers and convert to numpy array
        logger.info("1. Extracting embeddings from papers...")
        embeddings = [paper.get('embedding', []) for paper in papers]
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
        
        logger.info("\nHDBSCAN Results:")
        logger.info(f"  - Found {n_clusters} initial clusters")
        logger.info(f"  - {n_noise} unclustered papers ({(n_noise/len(papers))*100:.1f}% of total)")
        logger.info(f"  - {len(papers) - n_noise} clustered papers ({((len(papers) - n_noise)/len(papers))*100:.1f}% of total)")
        
        # Log individual cluster sizes
        cluster_sizes = {}
        for label in cluster_labels:
            if label != -1:
                cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
                
        logger.info("\nInitial Cluster Sizes:")
        for label, size in sorted(cluster_sizes.items()):
            logger.info(f"  - Cluster {label}: {size} papers ({(size/len(papers))*100:.1f}% of total)")
        
        # Group papers by cluster
        clusters_dict = {}
        uncategorized_papers = []
        
        # First, collect papers by cluster
        for idx, label in enumerate(cluster_labels):
            if label == -1:  # Noise points
                uncategorized_papers.append(papers[idx])
            else:
                if label not in clusters_dict:
                    clusters_dict[label] = []
                clusters_dict[label].append(papers[idx])
        
        # Process clusters in parallel
        tasks = []
        logger.info(f"\n4. Sending {len(clusters_dict)} clusters to Gemini for naming...")
        for label, cluster_papers in clusters_dict.items():
            tasks.append(get_cluster_task(cluster_papers))
            
        # Wait for all cluster naming tasks to complete
        cluster_names = await asyncio.gather(*tasks)
        
        # Create final clusters list
        final_clusters = []
        for (label, cluster_papers), task_name in zip(clusters_dict.items(), cluster_names):
            paper_ids = [p.get('id', '') for p in cluster_papers]
            final_clusters.append(Cluster(
                task=task_name,
                paper_ids=paper_ids,
                papers=cluster_papers
            ))
            
        # Add uncategorized papers if any exist
        if uncategorized_papers:
            final_clusters.append(Cluster(
                task="Uncategorized",
                paper_ids=[p.get('id', '') for p in uncategorized_papers],
                papers=uncategorized_papers
            ))
            
        # Log final clustering results
        logger.info("\nFinal Clustering Results:")
        logger.info(f"  - HDBSCAN identified: {n_clusters} clusters")
        logger.info(f"  - After Gemini naming: {len(final_clusters)} clusters")
        logger.info(f"  - Total papers: {len(papers)}")
        logger.info(f"  - Papers in clusters: {len(papers) - n_noise} ({((len(papers) - n_noise)/len(papers))*100:.1f}%)")
        logger.info(f"  - Unclustered papers: {n_noise} ({(n_noise/len(papers))*100:.1f}%)")
        
        logger.info("\nFinal Cluster Names and Sizes:")
        for cluster in sorted(final_clusters, key=lambda x: (-len(x.papers) if x.task != "Uncategorized" else -1)):
            logger.info(f"  - {cluster.task}: {len(cluster.papers)} papers ({(len(cluster.papers)/len(papers))*100:.1f}%)")
            
        logger.info(f"\n{'='*50}")
        logger.info("Clustering complete!")
        logger.info(f"{'='*50}\n")
            
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

async def summarize_cluster(task_name: str, papers: List[Dict[str, Any]], timeout: int = 30, retry_count=3, initial_delay=1) -> PaperSummary:
    """
    Generate a summary for a cluster of papers with retry logic.
    """
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
            
            response = await asyncio.wait_for(
                model.generate_content(prompt).async_result(),
                timeout=timeout
            )
            
            summary = PaperSummary(summary=response.text)
            return summary
        
        except TimeoutError:
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