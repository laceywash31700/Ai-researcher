from llama_index.core import Document, Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.indices.postprocessor import FixedRecencyPostprocessor
from llama_index.core.schema import IndexNode
from typing import List, Dict, Optional
from tools import fetch_from_arxiv, download_pdf, batch_download
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging

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
        Settings.chunk_overlap = 200

    def fetch_and_cache_papers(self, query: str, papers_count: int = 20, days_old: Optional[int] = None) -> None:
        """
        Fetch papers from arXiv and cache metadata
        
        Args:
            query: Search query
            max_results: Maximum papers to fetch
            days_old: Only fetch papers from last N days
        """
        try:
            self.papers = fetch_from_arxiv(query, papers_count, days_old)
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
            results = batch_download(
                downloadable,
                delay=delay
            )
            self._save_metadata()  # Update metadata with download status
            return results
        except Exception as e:
            logger.error(f"Failed to download papers: {str(e)}")
            raise

    def create_documents(self, include_pdf_text: bool = False) -> None:
        """
        Create enriched documents with optional PDF text extraction
        
        Args:
            include_pdf_text: Whether to extract text from downloaded PDFs
        """
        self.documents = []
        for paper in self.papers:
            # Base content from arXiv metadata
            content = (
                f"Title: {paper['title']}\n"
                f"Authors: {', '.join(paper['authors'])}\n"
                f"Abstract: {paper['summary']}\n"
                f"Published: {paper['published']}\n"
            )
            
            # Add PDF text if available and requested
            if include_pdf_text and paper.get('pdf_url'):
                pdf_path = self.pdf_dir / f"{paper['arxiv_id']}.pdf"
                if pdf_path.exists():
                    try:
                        pdf_text = self._extract_pdf_text(pdf_path)
                        content += f"\nFull Text Excerpts:\n{pdf_text[:10000]}"  # First 10k chars
                    except Exception as e:
                        logger.warning(f"Failed to extract PDF text: {str(e)}")
            
            doc = Document(
                text=content,
                metadata={
                    **{k: v for k, v in paper.items() if k not in ['summary', 'authors']},
                    "authors": ', '.join(paper['authors']),
                    "document_type": "research_paper",
                    "indexed_at": datetime.now().isoformat()
                },
                excluded_llm_metadata_keys=["pdf_url", "arxiv_url"]
            )
            self.documents.append(doc)

    def build_index(self, refresh: bool = False) -> None:
        """
        Build or refresh the vector index with advanced configuration
        
        Args:
            refresh: Whether to force rebuild even if index exists
        """
        if refresh or not self._index_exists():
            if not self.documents:
                raise ValueError("No documents available to index")
                
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
        """Save paper metadata to JSON"""
        with open(self.metadata_dir / "papers.json", "w") as f:
            json.dump(self.papers, f, indent=2)

    def _load_metadata(self) -> None:
        """Load paper metadata from JSON"""
        try:
            with open(self.metadata_dir / "papers.json", "r") as f:
                self.papers = json.load(f)
        except FileNotFoundError:
            self.papers = []

    def _index_exists(self) -> bool:
        """Check if index exists on disk"""
        return (self.index_dir / "docstore.json").exists()

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF using PyMuPDF or fallback"""
        try:
            import fitz  # PyMuPDF
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