import arxiv
import requests
import fitz
import pytesseract
from pdf2image import convert_from_path
import re
from typing import List, Dict, Any
from pathlib import Path
import logging
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


def extract_pdf_text(
    pdf_path: Path,
    use_ocr: bool = False,
    ocr_languages: str = 'eng'
) -> str:
    """
    Robust PDF text extraction with fallbacks:
    1. Try PyMuPDF (fastest)
    2. Try pdfplumber (better for some PDFs)
    3. Fallback to OCR if needed
    
    Args:
        pdf_path: Path to PDF file
        use_ocr: Force OCR even if text layers exist
        ocr_languages: Languages for Tesseract (e.g., 'eng+fra')
    
    Returns:
        Extracted text
    """
    text = ""
    
    # Method 1: PyMuPDF (fitz)
    if not use_ocr:
        try:
            with fitz.open(pdf_path) as doc:
                text = "\n".join(page.get_text() for page in doc)
            if text.strip():
                return text
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {str(e)}")
    
    # Method 2: pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(
                page.extract_text() 
                for page in pdf.pages 
                if page.extract_text()
            )
        if text.strip():
            return text
    except ImportError:
        logger.warning("pdfplumber not available")
    except Exception as e:
        logger.warning(f"pdfplumber failed: {str(e)}")
    
    # Method 3: OCR
    try:
        images = convert_from_path(pdf_path)
        text = "\n".join(
            pytesseract.image_to_string(
                image, 
                lang=ocr_languages
            ) 
            for image in images
        )
        return text
    except Exception as e:
        logger.error(f"OCR failed: {str(e)}")
        raise ValueError(f"All text extraction methods failed: {str(e)}")
    
def analyze_pdf_structure(pdf_path: Path) -> Dict:
    """
    Extract PDF metadata and structure:
    - Sections
    - Figures/Tables
    - Math content
    """
    try:
        text = extract_pdf_text(pdf_path)
        
        # Extract sections
        sections = {}
        for match in re.finditer(r'\n(\d+\.\s*[A-Z][^\n]+)', text):
            sections[match.group(1)] = match.start()
        
        # Count special elements
        stats = {
            'figures': text.lower().count('figure'),
            'tables': text.lower().count('table'),
            'equations': text.count('$') // 2,  # Rough estimate
            'pages': len(fitz.open(pdf_path))
        }
        
        return {
            'sections': sections,
            'stats': stats,
            'sample_text': text[:1000] + '...' if text else ''
        }
    except Exception as e:
        logger.error(f"PDF analysis failed: {str(e)}")
        return {'error': str(e)}

def clean_text(text: str) -> str:
    """Normalize text for LLM processing"""
    return " ".join(text.replace("\n", " ").replace("\t", " ").split())

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

