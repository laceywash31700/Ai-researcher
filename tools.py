import arxiv
import requests
import pytesseract
from pdf2image import convert_from_path
import re
from typing import List, Dict, Any
from pathlib import Path
import logging
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTFigure, LTRect
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import pytesseract
from tenacity import retry, stop_after_attempt, wait_exponential


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("research_tools.log"), 
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


client = arxiv.Client()


def _extract_arxiv_id(url: str) -> str:
    """
    Extracts the arXiv ID from a URL in any of these formats:
    - http://arxiv.org/abs/2504.13775v1
    - https://arxiv.org/pdf/2504.13775v1.pdf
    - http://arxiv.org/abs/2106.12345
    - https://arxiv.org/abs/math/0612345v2
    
    Returns:
        The arXiv ID (e.g., "2504.13775v1" or "math/0612345v2")
    """
    pattern = r'arxiv\.org/(?:abs|pdf)/(.+?)(?:\.pdf|$)'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract arXiv ID from URL: {url}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_from_arxiv(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch papers from arXiv with Pinecone-compatible metadata formatting.
    
    Args:
        query: Search query string
        max_results: Number of results to return (1-100)
        
    Returns:
        List of paper dictionaries with validated metadata
        
    Raises:
        ValueError: For invalid inputs or API errors
        RuntimeError: For temporary failures
    """
    # Input validation
    if not isinstance(query, str) or not query.strip():
        logger.error("Invalid query format")
        raise ValueError("Title must be a non-empty string")
        
    if not isinstance(max_results, int) or not (1 <= max_results <= 100):
        logger.error(f"Invalid max_results: {max_results}")
        raise ValueError("max_results must be an integer between 1-100")

    try:
        search = arxiv.Search(
            query=f'all:"{query}"',
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        papers = []
        for result in client.results(search):
            try:
                arxiv_id = _extract_arxiv_id(result.entry_id)
                
                # Convert all fields to Pinecone-compatible types
                paper_info = {
                    "arxiv_id": str(arxiv_id),
                    "title": str(result.title),
                    "summary": str(result.summary).replace('\n', ' ').strip(),
                    "published": result.published.isoformat(),  # Convert datetime to string
                    "journal_ref": str(result.journal_ref) if result.journal_ref else "",
                    "doi": str(result.doi) if result.doi else "",
                    "primary_category": str(result.primary_category),
                    "categories": list(result.categories),  # Ensure this is a list
                    "pdf_url": str(result.pdf_url),
                    "arxiv_url": str(result.entry_id),
                    "authors": [str(author.name) for author in result.authors],
                    "version": int(re.search(r'v(\d+)$', arxiv_id).group(1)) if 'v' in arxiv_id else 1,
                }
                
                # Remove empty fields that might cause issues
                paper_info = {k: v for k, v in paper_info.items() if v not in [None, ""]}
                papers.append(paper_info)
                
            except Exception as e:
                logger.warning(
                    f"Skipping paper {getattr(result, 'entry_id', 'unknown')}: {str(e)}",
                    exc_info=True
                )
                continue
            
        logger.info(f"Found {len(papers)} papers for query: '{query}'")
        return papers

    except arxiv.ArXivError as e:
        logger.error(f"arXiv API error: {str(e)}")
        raise ValueError(f"Search failed for '{query}'. Try different terms. Details: {str(e)}")
    except Exception as e:
        logger.exception("Unexpected arXiv fetch error")
        raise RuntimeError("Temporary search issue. Please try again later") from e

    
def download_pdf(pdf_url: str, output_file_name: str, storage_dir: Path = Path("storage/papers")) -> str:
    try:
        # Ensure storage directory exists
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        full_path = storage_dir / output_file_name

        if full_path.exists():
            return str(full_path.resolve())

        # Download with 10s timeout and retry once on failure
        for attempt in range(2):
            try:
                response = requests.get(pdf_url, timeout=10)
                response.raise_for_status()
                
                with open(full_path, "wb") as file:
                    file.write(response.content)
                    
                logger.info(f"Downloaded {output_file_name} to {full_path}")
                return str(full_path.resolve())
                
            except requests.exceptions.Timeout:
                if attempt == 0:
                    logger.warning(f"Timeout downloading {pdf_url}, retrying...")
                    continue
                raise
                
    except Exception as e:
        logger.error(f"PDF download failed: {str(e)}")
        # Clean up partially downloaded file if it exists
        if 'full_path' in locals() and full_path.exists():
            full_path.unlink()
        raise

def clean_text(text: str) -> str:
    """Clean extracted text."""
    text = re.sub(r'\s+', ' ', text)  # Remove excessive whitespace
    return text.strip()

def extract_pdf_text(pdf_path: str, use_ocr: bool = False) -> str:
    """Extract text from PDF, with optional OCR fallback."""
    full_text = []
    
    # First try standard text extraction
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text.append(clean_text(text))
    except Exception as e:
        print(f"Standard extraction failed: {e}")
    
    # If no text found or OCR requested, use OCR
    if not full_text or use_ocr:
        images = convert_from_path(pdf_path)
        for image in images:
            text = pytesseract.image_to_string(image)
            full_text.append(clean_text(text))
    
    return "\n".join(full_text)
        
def analyze_pdf_structure(pdf_path: Path) -> Dict:
    """
    Extract PDF structure using PyPDF2 and pdfminer.six
    """
    try:
        # Get text content using existing extract_text_from_pdf function
        text = extract_pdf_text(str(pdf_path))
        
        # Initialize PDF reader
        reader = PdfReader(pdf_path)
        
        # Extract document structure
        sections = {}
        figures = 0
        tables = 0
        equations = 0
        
        for i, page in enumerate(extract_pages(str(pdf_path))):
            # Extract sections from text
            if i == 0:  # First page analysis
                for match in re.finditer(r'\n(\d+\.\s*[A-Z][^\n]+)', text):
                    sections[match.group(1)] = match.start()
            
            # Analyze page layout
            for element in page:
                if isinstance(element, LTTextContainer):
                    # Detect equations (basic pattern matching)
                    if '$' in element.get_text():
                        equations += 1
                elif isinstance(element, LTFigure):
                    figures += 1
                elif isinstance(element, LTRect):
                    tables += 1  # Simple table detection

        return {
            'sections': sections,
            'stats': {
                'figures': figures,
                'tables': tables,
                'equations': equations,
                'pages': len(reader.pages)
            },
            'sample_text': text[:1000] + '...' if text else ''
        }
        
    except Exception as e:
        logger.error(f"PDF analysis failed: {str(e)}")
        return {'error': str(e)}



def format_arxiv_results(papers: List[Dict], query: str) -> str:
    """Convert raw arXiv results to clean Markdown output"""
    if not papers:
        return f"No papers found for '{query}'. Try broadening your search terms."

    # Filter for relevant papers (simple keyword match)
    relevant_papers = [
        p
        for p in papers
        if any(
            term.lower() in p["title"].lower() or term.lower() in p["summary"].lower()
            for term in query.split()
        )
    ][
        :5
    ]  # Show max 5 most relevant

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

