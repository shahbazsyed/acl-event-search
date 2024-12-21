import requests
from bs4 import BeautifulSoup
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import pickle
import hashlib
from typing import Dict, List, Optional, Tuple

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

class EventIndex:
    def __init__(self, event_url: str):
        self.event_url = event_url
        self.papers = []
        self.embeddings = None
        
        # Create a unique identifier for this event
        self.event_id = hashlib.md5(event_url.encode()).hexdigest()
        
        # Set up cache paths
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.papers_cache = os.path.join(self.cache_dir, f"papers_{self.event_id}.json")
        self.embeddings_cache = os.path.join(self.cache_dir, f"embeddings_{self.event_id}.pkl")
    
    def load_cache(self) -> bool:
        """Load cached papers and embeddings if they exist"""
        try:
            if os.path.exists(self.papers_cache) and os.path.exists(self.embeddings_cache):
                with open(self.papers_cache, 'r') as f:
                    import json
                    self.papers = json.load(f)
                with open(self.embeddings_cache, 'rb') as f:
                    self.embeddings = pickle.load(f)
                return True
        except Exception as e:
            print(f"Error loading cache: {e}")
        return False
    
    def save_cache(self):
        """Save papers and embeddings to cache"""
        try:
            with open(self.papers_cache, 'w') as f:
                import json
                json.dump(self.papers, f)
            with open(self.embeddings_cache, 'wb') as f:
                pickle.dump(self.embeddings, f)
        except Exception as e:
            print(f"Error saving cache: {e}")

def parse_html(html_content, base_url):
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

    return papers

def get_paper_info(url: str) -> Optional[EventIndex]:
    """Fetch and parse paper information from a given URL"""
    try:
        # Create or load existing index
        event_index = EventIndex(url)
        
        # Try to load from cache first
        if event_index.load_cache():
            return event_index
        
        # If not in cache, fetch and process
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text

        # Extract base URL
        base_url = re.match(r"(https?://[^/]+)", url).group(1)
        
        # Parse papers
        papers = parse_html(html_content, base_url)
        
        # Log the number of papers with abstracts
        total_papers = len(papers)
        print(f"Found {total_papers} papers with abstracts (excluding proceedings)")
        
        event_index.papers = papers
        
        # Generate embeddings
        event_index.embeddings = initialize_embeddings(event_index.papers)
        
        # Cache the results
        event_index.save_cache()
        
        return event_index

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None
    except AttributeError:
        print(f"Error extracting base URL from: {url}")
        return None

def get_text_for_embedding(paper):
    """Combine title and abstract for embedding"""
    title = paper.get('title', '')
    abstract = paper.get('abstract', '')
    if abstract:
        return f"{title} {abstract}"
    return title

def initialize_embeddings(papers: List[dict]) -> np.ndarray:
    """Initialize embeddings for papers"""
    texts = [get_text_for_embedding(paper) for paper in papers]
    return model.encode(texts, convert_to_tensor=False)

def search_papers(query: str, event_index: EventIndex, top_k: int = 10) -> List[dict]:
    """Search for papers using semantic similarity within a specific event index"""
    if not event_index.papers or event_index.embeddings is None:
        return []
    
    # Encode the query
    query_embedding = model.encode(query, convert_to_tensor=False)
    
    # Calculate similarities
    similarities = np.dot(event_index.embeddings, query_embedding)
    
    # Get top k results
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Format results
    results = []
    for idx in top_indices:
        paper = event_index.papers[idx]
        results.append({
            "title": paper["title"],
            "authors": paper["authors"],
            "abstract": paper["abstract"],
            "link": paper["link"],
            "score": round(float(similarities[idx]), 3)
        })
    
    return results
