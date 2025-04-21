from llama_index.core import Document, Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.indices.postprocessor import FixedRecencyPostprocessor
from typing import List, Dict, Optional
from tools import fetch_from_arxiv, download_pdf
from pathlib import Path
import json
import re
from datetime import datetime
import logging
import fitz 

logger = logging.getLogger(__name__)

class IndexManager:
    def __init__(self, embedding_model, persist_dir: str = "storage"):
        """
        Enhanced index manager with semantic chunking, hybrid search, and incremental updates
        
        Args:
            embedding_model: Embedding model to use
            persist_dir: Directory to store indices and metadata
        """
        self.embedding_model = embedding_model
        self.persist_dir = Path(persist_dir)
        self.papers: List[Dict] = []
        self.documents: List[Document] = []
        self.index = None
        
        # Configure storage paths
        self.index_dir = self.persist_dir / "index"
        self.metadata_dir = self.persist_dir / "metadata"
        self.pdf_dir = self.persist_dir / "papers"
        
        # Create directories if they don't exist
        for directory in [self.index_dir, self.metadata_dir, self.pdf_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Configure advanced node parsing
        self.node_parser = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=self.embedding_model
        )
        
        Settings.embed_model = self.embedding_model
        Settings.chunk_size = 1024
        Settings.chunk_overlap = 50

    def fetch_and_cache_papers(self, query: str, max_results: int = 10) -> None:
        """
        Fetch papers from arXiv and cache metadata
        
        Args:
            query: Search query
            max_results: Maximum papers to fetch
        """
        try:
            self.papers = fetch_from_arxiv(query, max_results)
            self._save_metadata()
            logger.info(f"Fetched and cached {len(self.papers)} papers")
        except Exception as e:
            logger.error(f"Failed to fetch papers: {str(e)}")
            raise

    def download_papers(self, max_downloads: int = 5, delay: float = 2.0) -> Dict[str, str]:
        """
        Download PDFs for cached papers with rate limiting
        
        Args:
            max_downloads: Maximum PDFs to download
            delay: Seconds between downloads
            
        Returns:
            Dictionary of download results
        """
        try:
            downloadable = [p for p in self.papers if p.get('downloadable')][:max_downloads]
            results = download_pdf(
                downloadable,
                delay=delay
            )
            self._save_metadata()
            return results
        except Exception as e:
            logger.error(f"Failed to download papers: {str(e)}")
            raise

    def create_documents(self, include_pdf_text: bool = True) -> None:
        """
        Enhanced document creator with robust PDF text extraction
        """
        self.documents = []
        for paper in self.papers:
            # Base metadata
            metadata = {
                **{k: v for k, v in paper.items() if k not in ['summary', 'authors']},
                "authors": ', '.join(paper['authors']),
                "document_type": "research_paper",
                "indexed_at": datetime.now().isoformat()
            }
            
            # 1. Always include abstract and metadata
            content_parts = [
                f"Title: {paper['title']}",
                f"Authors: {metadata['authors']}",
                f"Abstract: {paper['summary']}",
                f"Published: {paper['published']}"
            ]
            
            # 2. Add full PDF text if available
            pdf_text = ""
            if include_pdf_text and paper.get('pdf_url'):
                pdf_path = self.pdf_dir / f"{paper['arxiv_id']}.pdf"
                if pdf_path.exists():
                    try:
                        pdf_text = self._extract_pdf_text(pdf_path)
                        content_parts.append(f"\nFull Text:\n{pdf_text}")
                        
                        # Add extracted sections to metadata
                        metadata.update(self._extract_pdf_metadata(pdf_text))
                    except Exception as e:
                        logger.warning(f"PDF extraction failed for {pdf_path}: {str(e)}")
                        content_parts.append(f"\n[PDF extraction failed: {str(e)}]")
            
            doc = Document(
                text="\n".join(content_parts),
                metadata=metadata,
                excluded_llm_metadata_keys=["pdf_url", "arxiv_url"]
            )
            self.documents.append(doc)
            
    def _extract_pdf_metadata(self, pdf_text: str) -> Dict[str, str]:
        """Extract key sections from PDF text for enhanced metadata"""
        metadata = {}
        
        # Extract key sections using regex
        introduction_match = re.search(r'(?i)\n(1|i)\.?\s*introduction(.+?)(\n(2|ii)\.|\Z)', 
                                     pdf_text, re.DOTALL)
        if introduction_match:
            metadata["introduction_excerpt"] = introduction_match.group(2)[:500] + "..."
            
        conclusion_match = re.search(r'(?i)\n(7|vii)\.?\s*conclusion(.+?)(\nreferences|\Z)', 
                                   pdf_text, re.DOTALL)
        if conclusion_match:
            metadata["conclusion_excerpt"] = conclusion_match.group(2)[:500] + "..."
            
        # Count figures/tables
        metadata["figure_count"] = pdf_text.lower().count("figure")
        metadata["table_count"] = pdf_text.lower().count("table")
        
        return metadata
    
    
    def build_index(self, refresh: bool = False) -> None:
        """
        Build or refresh the vector index with advanced configuration
        
        Args:
            refresh: Whether to force rebuild even if index exists
        """
        if refresh or not self._index_exists():
            if not self.documents:
            # Try loading existing papers first
                self._load_metadata()
            if self.papers:
                self.create_documents()
            else:
                raise ValueError(
                    "No documents available. "
                    "Call fetch_and_cache_papers() first or check storage."
                )
                
            self.index = VectorStoreIndex.from_documents(
                self.documents,
                transformations=[self.node_parser],
                show_progress=True
            )
            
            # Configure advanced retrieval
            self.index._node_postprocessors = [
                SimilarityPostprocessor(similarity_cutoff=0.7),
                KeywordNodePostprocessor(required_keywords=["study", "research", "result"]),
                FixedRecencyPostprocessor(
                    top_k=1,
                    date_key="published",
                    storage_context=self.index.storage_context
                )
            ]
            
            # Enable auto-merging of chunks
            self.index._retriever = AutoMergingRetriever(
                self.index.as_retriever(similarity_top_k=6),
                self.index.storage_context,
                verbose=True
            )
            
            self._persist_index()
        else:
            self.load_index()
            
            
    def load_index(self) -> VectorStoreIndex:
        """Smart index loader that auto-initializes if needed"""
        try:
            # First check if we need to create a default index
            if not self._index_exists():
                logger.info("No existing index found - creating default index...")
                self.fetch_and_cache_papers("machine learning", papers_count=10)
                self.create_documents()
                self.build_index()
            
            # Proceed with normal load
            storage_context = StorageContext.from_defaults(
                persist_dir=str(self.index_dir)
            )  
             
            self.index = load_index_from_storage(
                storage_context,
                transformations=[self.node_parser],
                embed_model=self.embedding_model
            )
            return self.index
        
        except Exception as e:
            logger.error(f"Index loading failed: {str(e)}")
            raise RuntimeError(f"Could not initialize index: {str(e)}")
        

    def get_paper_by_id(self, arxiv_id: str) -> Optional[Dict]:
        """Retrieve paper metadata by arXiv ID"""
        return next((p for p in self.papers if p.get('arxiv_id') == arxiv_id), None)
    
    def list_papers(self) -> List[Dict]:
        for paper in self.papers:
            print(f"Title: {paper['title']}, Authors: {', '.join(paper['authors'])}")

    def search_papers(self, query: str, **kwargs) -> List[Dict]:
        """
        Search papers in local cache
        
        Args:
            query: Search string
            kwargs: Additional filters (e.g., year=2023)
            
        Returns:
            List of matching papers
        """
        results = []
        for paper in self.papers:
            match = True
            # Text search
            if query.lower() not in paper['title'].lower() and \
               query.lower() not in paper['summary'].lower():
                continue
                
            # Filter checks
            for k, v in kwargs.items():
                if str(paper.get(k)).lower() != str(v).lower():
                    match = False
                    break
                    
            if match:
                results.append(paper)
                
        return results

    # Private helper methods
    def _persist_index(self) -> None:
        """Persist index and metadata"""
        self.index.storage_context.persist(persist_dir=str(self.index_dir))
        self._save_metadata()
        logger.info("Persisted index and metadata")

    def _save_metadata(self) -> None:
        """Save paper metadata to JSON with datetime handling"""
        def default_serializer(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        with open(self.metadata_dir / "papers.json", "w") as file:
            json.dump(self.papers, file, indent=2, default=default_serializer)


    def _load_metadata(self) -> None:
        """Load paper metadata from JSON"""
        try:
            with open(self.metadata_dir / "papers.json", "r") as file:
                self.papers = json.load(file)
        except FileNotFoundError:
            self.papers = []

    def _index_exists(self) -> bool:
        """Check if index exists on disk"""
        return (self.index_dir / "docstore.json").exists()

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF using PyMuPDF or fallback"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except ImportError:
            logger.warning("PyMuPDF not installed, using slower fallback")
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            return "\n".join(page.extract_text() for page in reader.pages)
    
    