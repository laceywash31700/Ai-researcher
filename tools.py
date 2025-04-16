import arxiv
import requests

from typing import List, Dict, Optional
from datetime import datetime, timedelta 
import time
from pathlib import Path
import logging
from tenacity import retry, stop_after_attempt, wait_exponential 


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("research_tools.log"),  # Save to file
        logging.StreamHandler()  # Print to console
    ]
)

logger = logging.getLogger(__name__)

# Configure arXiv client with enhanced settings
client = arxiv.Client(
    page_size=100,
    delay_seconds=5,
    num_retries=8
)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_from_arxiv(query: str, max_results: int = 10, days_old: Optional[int] = None) -> List[Dict]:
    """
    Robust arXiv paper fetcher with complete field handling
    Args:
        query: Search query
        max_results: Number of papers to return (1-1000)
        days_old: Only return papers from last N days
    Returns:
        List of paper dictionaries with all available metadata
    """
    try:
        # Build date-filtered query if specified
        if days_old:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            query = f"{query} AND submittedDate:[{cutoff_date.strftime('%Y%m%d')} TO *]"
        
        search = arxiv.Search(
            query=query,
            max_results=min(max_results, 1000),  # arXiv's max limit
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for result in client.results(search):
            try:
                paper = {
                    "entry_id": result.entry_id,
                    "title": result.title,
                    "summary": clean_text(result.summary),
                    "published": result.published.isoformat(),
                    "updated": result.updated.isoformat(),
                    "authors": [author.name for author in result.authors],
                    "comment": result.comment or "",
                    "journal_ref": result.journal_ref or "",
                    "doi": result.doi or "",
                    "primary_category": result.primary_category,
                    "categories": result.categories,
                    "links": [link.href for link in result.links],
                    "pdf_url": next((link.href for link in result.links 
                                  if link.title == "pdf"), None),
                    "arxiv_url": f"https://arxiv.org/abs/{result.entry_id.split('/')[-1]}",
                    "downloadable": any(link.title == "pdf" for link in result.links)
                }
                papers.append(paper)
            except Exception as e:
                logger.warning(f"Error processing paper {result.entry_id}: {str(e)}")
                continue
                
        return papers
        
    except arxiv.ArXivError as e:
        logger.error(f"arXiv API error: {str(e)}")
        raise ValueError(f"arXiv API request failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise RuntimeError(f"Paper fetching failed: {str(e)}")
      
def download_pdf(pdf_url: str, output_file_name: Optional[str] = None, timeout: int = 45) -> str:
    """
    Robust PDF downloader with:
    - Automatic arXiv ID filename
    - Timeout handling
    - Duplicate detection
    """
    try:
        papers_dir = Path("papers")
        papers_dir.mkdir(exist_ok=True)
        
        if not output_file_name:
            arxiv_id = pdf_url.split('/')[-1].replace('.pdf', '')
            output_file_name = f"{arxiv_id}.pdf"

        full_path = papers_dir / output_file_name
        
        if full_path.exists():
            return str(full_path.resolve())

        response = requests.get(
            pdf_url,
            stream=True,
            timeout=timeout,
            headers={'User-Agent': 'ResearchBot/1.0'}
        )
        response.raise_for_status()

        with open(full_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return str(full_path.resolve())

    except Exception as e:
        logger.error(f"PDF download failed: {str(e)}")
        raise

def clean_text(text: str) -> str:
    """Normalize text for LLM processing"""
    return ' '.join(text.replace('\n', ' ').replace('\t', ' ').split())

# Additional utilities
def get_recent_papers(keyword: str, last_n_days: int = 30, max_papers: int = 10) -> List[Dict]:
    return fetch_from_arxiv(
        f'ti:"{keyword}" OR abs:"{keyword}"',  # Search title AND abstract
        paper_count=max_papers,
        days_old=last_n_days
    )

def batch_download(papers: List[Dict], max_workers: int = 3, delay: float = 3.0) -> Dict[str, str]:
    """
    Parallel PDF downloader with:
    - Rate limiting
    - Thread safety
    """
    from concurrent.futures import ThreadPoolExecutor
    results = {}
    
    def _download(paper):
        try:
            if paper.get('pdf_url'):
                time.sleep(delay)
                return paper['arxiv_id'], download_pdf(paper['pdf_url'])
        except Exception as e:
            logger.warning(f"Failed to download {paper.get('arxiv_id')}: {str(e)}")
            return paper.get('arxiv_id'), str(e)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for arxiv_id, result in executor.map(_download, papers):
            if arxiv_id:
                results[arxiv_id] = result
    return results
  
def format_arxiv_results(papers: List[Dict], query: str) -> str:
    """Convert raw arXiv results to clean Markdown output"""
    if not papers:
        return f"No papers found for '{query}'. Try broadening your search terms."
    
    # Filter for relevant papers (simple keyword match)
    relevant_papers = [
        p for p in papers 
        if any(term.lower() in p['title'].lower() or 
               term.lower() in p['summary'].lower()
               for term in query.split())
    ][:5]  # Show max 5 most relevant
    
    if not relevant_papers:
        return f"No papers directly matching '{query}' found. Try different keywords."

    formatted = [
        f"### {i}. [{p['title']}]({p['arxiv_url']})\n"
        f"**Authors**: {', '.join(p['authors'][:3])}{' et al.' if len(p['authors']) > 3 else ''}\n"
        f"**Published**: {p['published'][:10]}\n"
        f"**Categories**: {', '.join(p['categories'])}\n"
        f"{'[📥 PDF Available](' + p['pdf_url'] + ')' if p['downloadable'] else ''}\n\n"
        f"{p['summary'][:250].replace('\n', ' ')}...\n"
        for i, p in enumerate(relevant_papers, 1)
    ]
    return "\n\n".join(formatted)