from constants import PINECONE_API_KEY
from index_manager import IndexManager
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from typing import List, Dict

import logging

logger = logging.getLogger(__name__)


class IndexManagerPinecone(IndexManager):
    def __init__(
        self, 
        embedding_model, 
        index_name: str, 
    ):
        """
        Pinecone-specific index manager with enhanced capabilities
        
        Args:
            embedding_model: Embedding model to use
            index_name: Name of Pinecone index
        """
        super().__init__(embedding_model)
        self.documents = []
        # Initialize Pinecone client
        self.index_name = index_name
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        # Configure index parameters and Initialize Pinecone index
        self.pinecone_index = self.pc.Index(self.index_name)
        # Set up vector store
        self.vector_store = PineconeVectorStore(
            pinecone_index=self.pinecone_index
        )
        
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        self.index = None

    def create_index(self):
        """
        Create Pinecone index with documents
        """
        self.create_documents()
        # Configure settings
        Settings.chunk_size = 1024
        Settings.chunk_overlap = 50

        # Create index
        self.index = VectorStoreIndex.from_documents(
            self.documents,
            storage_context=self.storage_context,
            embed_model=self.embedding_model,
            show_progress=True
        )

    def retrieve_index(self):
        """Retrieve existing Pinecone index"""
        if not self.index:
         self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store, 
            embed_model=self.embedding_model,
            show_progress=True,
        )
        return self.index
        
    
    def upsert_to_index(self, papers: List[Dict]):
        try:
            # 1. Add new papers to local cache
            self.papers.extend(papers)
            
            # 2. Create documents for NEW PAPERS ONLY
            self.create_documents()
            
            # 3. Process ONLY NEW DOCUMENTS
            new_nodes = []
            for doc in self.documents[-len(papers):]:  # Only process new docs
                # Generate node from document
                node = self.node_parser.get_nodes_from_documents([doc])[0]
                
                # MANUALLY SET EMBEDDING
                node.embedding = self.embedding_model.get_text_embedding(
                    node.get_content()
                )
                new_nodes.append(node)
            
            # 4. Add ONLY NEW NODES to Pinecone
            self.vector_store.add(new_nodes)  # CORRECT: Use manually embedded nodes
            
            return True
            
        except Exception as e:
            logger.error(f"Upsert failed: {str(e)}")
            return False