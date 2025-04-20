import arxiv
import requests
import fitz
import pytesseract
from pdf2image import convert_from_path
import re
import time 
from typing import List, Dict
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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_from_arxiv(title: str, max_results: int = 10):
    if not isinstance(title, str) or not title.strip():
        logger.error("Invalid title format")
        raise ValueError("Title must be a non-empty string")
    if not isinstance(max_results, int) or max_results <= 0:
        logger.error(f"Invalid max_results: {max_results}")
        raise ValueError("max_results must be a positive integer")

    try:
        search_query = f'all:"{title}"'
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        papers = []
        for result in client.results(search):
            try:
                paper_info = {
                    "title": result.title,
                    "summary": result.summary,
                    "published": result.published,
                    "journal_ref": result.journal_ref,
                    "doi": result.doi,
                    "primary_category": result.primary_category,
                    "categories": result.categories,
                    "pdf_url": result.pdf_url,
                    "arxiv_url": result.entry_id,
                    "authors": [author.name for author in result.authors],
                }
                papers.append(paper_info)
            except Exception as e:
                logger.warning(
                    f"Error processing paper {getattr(result, 'entry_id', 'unknown')}: {str(e)}"
                )
                continue
            
        print(papers)
        if not papers:
            logger.warning(f"No valid papers found for query: {title}")
        return papers

    except arxiv.ArXivError as e:
        logger.error(f"arXiv API error: {str(e)}")
        raise ValueError(f"arXiv search failed. Please try different terms. Technical details: {str(e)}")
    except Exception as e:
        logger.exception("Unexpected arXiv fetch error")  
        raise RuntimeError(f"Temporary system issue. Please try again later. Reference: {id(e)}") from e


    
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

