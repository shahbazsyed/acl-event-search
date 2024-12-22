import requests
from bs4 import BeautifulSoup
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import pickle
import hashlib
import aiohttp
import asyncio
import logging
import json
import torch
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings("ignore")

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

class EventIndex:
    def __init__(self, papers: List[Dict], event_url: str):
        if not papers:
            raise ValueError("Cannot create EventIndex with empty papers list")
            
        # Store event URL for identification
        self.event_url = event_url
        
        # Create cache directory if it doesn't exist
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.papers = papers
        self.embeddings = None
        self.initialize_embeddings()
    
    def _get_cache_key(self) -> str:
        """Generate a unique cache key based on event URL and paper contents"""
        # Include event URL in hash to ensure different events don't share cache
        url_hash = hashlib.md5(self.event_url.encode()).hexdigest()[:8]
        # Sort paper IDs to ensure consistent cache key
        paper_ids = sorted([p.get('id', '') for p in self.papers])
        content_hash = hashlib.md5(''.join(paper_ids).encode()).hexdigest()[:8]
        return f"{url_hash}_{content_hash}"
    
    def _get_cache_paths(self) -> Tuple[str, str]:
        """Get cache file paths for papers and embeddings"""
        cache_key = self._get_cache_key()
        papers_cache = os.path.join(self.cache_dir, f"papers_{cache_key}.json")
        embeddings_cache = os.path.join(self.cache_dir, f"embeddings_{cache_key}.pkl")
        return papers_cache, embeddings_cache
    
    def load_cache(self) -> bool:
        """Load cached papers and embeddings if they exist"""
        papers_cache, embeddings_cache = self._get_cache_paths()
        
        try:
            if os.path.exists(papers_cache) and os.path.exists(embeddings_cache):
                logger.info(f"Found cached papers and embeddings for event: {self.event_url}")
                with open(papers_cache, 'r') as f:
                    self.papers = json.load(f)
                with open(embeddings_cache, 'rb') as f:
                    self.embeddings = pickle.load(f)
                return True
        except Exception as e:
            logger.error(f"Error loading cache for event {self.event_url}: {e}")
        return False
    
    def save_cache(self):
        """Save papers and embeddings to cache"""
        papers_cache, embeddings_cache = self._get_cache_paths()
        
        try:
            logger.info(f"Saving papers and embeddings to cache for event: {self.event_url}")
            with open(papers_cache, 'w') as f:
                json.dump(self.papers, f)
            with open(embeddings_cache, 'wb') as f:
                pickle.dump(self.embeddings, f)
        except Exception as e:
            logger.error(f"Error saving cache for event {self.event_url}: {e}")
    
    def initialize_embeddings(self):
        """Initialize embeddings for papers."""
        try:
            # Try to load from cache first
            if self.load_cache():
                logger.info("Successfully loaded cached embeddings")
                return
            
            logger.info("Generating embeddings for papers")
            # Generate embeddings for all papers
            texts = [get_text_for_embedding(p) for p in self.papers]
            self.embeddings = model.encode(texts, convert_to_tensor=True)
        
            # Save to cache
            self.save_cache()
            logger.info("Successfully generated and cached embeddings")
        
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            raise

def get_text_for_embedding(paper: Dict) -> str:
    """Combine title and abstract for embedding."""
    title = paper.get('title', '').strip()
    abstract = paper.get('abstract', '').strip()
    return f"{title}\n\n{abstract}"

def parse_html(html_content: str, base_url: str) -> List[Dict]:
    """Parse HTML content to extract paper information"""
    soup = BeautifulSoup(html_content, 'html.parser')
    papers = []

    for p_tag in soup.find_all('p', class_='d-sm-flex align-items-stretch'):
        paper = {}

        # Title and link
        title_link = p_tag.find('a', href=re.compile(r'/\d{4}\.\w+-\w+\.\d+/$'))
        if title_link:
            title = title_link.text
            # Skip proceedings papers
            if title.startswith('Proceedings of '):
                continue
            paper['title'] = title
            paper['link'] = base_url + "/" + title_link['href'][1:]
            paper['id'] = title_link['href'].split('/')[-2]  # Get ID from URL

        # Authors
        authors = []
        for author_link in p_tag.find_all('a', href=re.compile(r'/people/')):
            authors.append(author_link.text)
        paper['authors'] = authors

        # Abstract
        abs_link = p_tag.find('a', href=re.compile(r'#abstract'))
        if abs_link:
            abstract_id = abs_link['href']
            abstract_div = soup.find('div', id=abstract_id[1:])
            if abstract_div:
                paper['abstract'] = abstract_div.find('div', class_='card-body').text.strip()
            else:
                paper['abstract'] = None
        else:
            paper['abstract'] = None

        # Only include papers with title and abstract
        if "title" in paper and paper['abstract']:
            papers.append(paper)

    logger.info(f"Found {len(papers)} papers with abstracts (excluding proceedings)")
    return papers

async def get_paper_info(url: str) -> List[Dict]:
    """Fetch and parse paper information from a given URL"""
    # Create cache directory if it doesn't exist
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key from URL
    url_hash = hashlib.md5(url.encode()).hexdigest()
    papers_cache = os.path.join(cache_dir, f"raw_papers_{url_hash}.json")
    
    # Try to load from cache first
    try:
        if os.path.exists(papers_cache):
            logger.info("Loading papers from cache")
            with open(papers_cache, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading papers cache: {e}")
    
    try:
        # Fetch HTML content
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to fetch URL: {url}")
                html_content = await response.text()

        # Extract base URL
        base_url_match = re.match(r"(https?://[^/]+)", url)
        if not base_url_match:
            raise ValueError(f"Could not extract base URL from: {url}")
        base_url = base_url_match.group(1)
        
        # Parse papers
        papers = parse_html(html_content, base_url)
        
        if not papers:
            logger.warning("No papers found with abstracts")
            return []
        
        # Save to cache
        try:
            logger.info(f"Saving {len(papers)} papers to cache")
            with open(papers_cache, 'w') as f:
                json.dump(papers, f)
        except Exception as e:
            logger.error(f"Error saving papers cache: {e}")
        
        return papers

    except aiohttp.ClientError as e:
        logger.error(f"Error fetching URL: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing papers: {e}")
        raise

async def search_papers(event_index: EventIndex, query: str, top_k: int = 10) -> List[Dict]:
    """Search papers using cosine similarity with query embedding."""
    try:
        # Get query embedding
        query_embedding = model.encode(query, convert_to_tensor=True)
        
        # Calculate similarities for all papers
        similarities = torch.matmul(event_index.embeddings, query_embedding)
        
        # Get top k results
        top_indices = torch.argsort(similarities, descending=True)[:top_k].cpu().numpy()
        
        # Format results with scores
        results = []
        for idx in top_indices:
            paper = event_index.papers[idx]
            score = float(similarities[idx])
            results.append({
                "id": paper.get("id"),
                "title": paper["title"],
                "authors": paper.get("authors", []),
                "abstract": paper.get("abstract", ""),
                "url": paper.get("url", ""),
                "score": score
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Error in search_papers: {e}")
        raise

async def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text using sentence-transformers."""
    try:
        # Truncate text to fit within model's limit if needed
        text = text[:512]  # Adjust based on model's requirements
        
        # Convert the text to embedding using the sentence transformer model
        with torch.no_grad():  # No need to track gradients for inference
            embedding = model.encode(text)
            
        # Convert to list for JSON serialization
        return embedding.tolist()
        
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
        raise

async def load_papers_from_url(url: str) -> List[Dict[str, Any]]:
    """Load papers from an ACL Anthology event URL."""
    try:
        # Fetch and parse papers
        papers = await get_paper_info(url)
        if not papers:
            raise ValueError("No papers found")
            
        logger.info(f"Successfully fetched {len(papers)} papers")
        return papers
        
    except Exception as e:
        logger.error(f"Error loading papers from URL: {str(e)}", exc_info=True)
        raise
