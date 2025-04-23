from typing import List, Dict
import logging
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from tools import fetch_from_arxiv

logger = logging.getLogger(__name__)

class IndexManager:
    def __init__(self, embedding_model):
        """
        Enhanced index manager with semantic chunking and hybrid search
        """
        self.embedding_model = embedding_model
        self.papers: List[Dict] = []
        self.documents: List[Document] = []
        self.index = None
        
        
        # Configure advanced node parsing
        self.node_parser = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=self.embedding_model
        )
        

    def fetch_and_cache_papers(self, topic: str, max_results: int = 10) -> None:
        """
        Fetch papers from arXiv and cache metadata with enhanced error handling
        """
        try:
            logger.info(f"Fetching papers for topic: {topic}")
            self.papers = fetch_from_arxiv(topic, max_results)
            
            logger.info(f"Successfully fetched and cached {len(self.papers)} papers")
        except Exception as e:
            logger.error(f"Failed to fetch papers: {str(e)}", exc_info=True)
            raise RuntimeError(f"Paper fetching failed: {str(e)}") from e

    def create_documents(self) -> None:
        """
        Enhanced document creator with PDF text extraction and metadata enrichment
        """
        for paper in self.papers:
            try:
                # Base metadata
                metadata = {
                    "arxiv_id": paper.get('arxiv_id', ''),
                    "title": paper.get('title', ''),
                    "published": paper.get('published', ''),
                    "authors": ", ".join(paper.get('authors', [])),
                    "categories": ", ".join(paper.get('categories', [])),
                    "pdf_url": paper.get('pdf_url', '')
                }
                
                # Base content
                content_parts = [
                    f"Title: {paper['title']}",
                    f"Authors: {', '.join(paper['authors'])}",
                    f"Published: {paper['published']}",
                    f"Summary: {paper['summary']}",
                    f"Categories: {', '.join(paper['categories'])}"
                ]
                
                
                # Create document with metadata
                doc = Document(
                    text="\n\n".join(content_parts),
                    metadata=metadata,
                    metadata_separator="::"
                )
                self.documents.append(doc)
                
            except Exception as e:
                logger.error(f"Failed to create document for paper {paper.get('arxiv_id')}: {str(e)}")
                continue

    
    def list_papers(self) -> List[Dict]:
        for paper in self.papers:
            print(f"Title: {paper['title']}, Authors: {', '.join(paper['authors'])}")

