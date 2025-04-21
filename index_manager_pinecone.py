import os

from constants import PINECONE_API_KEY
from pinecone import Pinecone

from index_manager import IndexManager
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from tools import logger



class IndexManagerPinecone(IndexManager):
    def __init__(self, embedding_model, index_name, persist_dir: str = "storage"):
        super().__init__(embedding_model, persist_dir)
        pc = Pinecone(api_key=PINECONE_API_KEY)
        self.pinecone_index = pc.Index(index_name)
        self.vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

    def create_index(self):
        self.documents = []
        self.create_documents()
        Settings.chunk_size = 1024
        Settings.chunk_overlap = 50
        VectorStoreIndex.from_documents(
            self.documents,
            storage_context=self.storage_context,
            embed_model=self.embedding_model,
        )

    def retrieve_index(self):
        return VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store, embed_model=self.embedding_model
        )
    
    def is_index_connected(self) -> bool:
        """Verify the index is properly connected and accessible"""
        try:
            # Check basic index stats
            stats = self.pinecone_index.describe_index_stats()
            if not stats:
                return False
                
            # Verify we can query the index
            test_query = [0.0] * self.embedding_model.dimension  # Null vector
            self.pinecone_index.query(vector=test_query, top_k=1)
            return True
        except Exception as e:
            logger.error(f"Index connection check failed: {str(e)}")
            return False
