import os
import logging
from typing import Dict, List, Optional, Any
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class VectorStoreService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self._initialize_pinecone()
            self._initialize_embeddings()
            self._initialize_vector_store()
            self.initialized = True

    def _initialize_pinecone(self):
        """Initialize Pinecone client and index"""
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "")

        if not all([self.pinecone_api_key, self.pinecone_environment, self.pinecone_index_name]):
            logger.error("Missing required Pinecone environment variables.")
            raise EnvironmentError(
                "Ensure PINECONE_API_KEY, PINECONE_ENVIRONMENT, and PINECONE_INDEX_NAME are set."
            )

        try:
            self.pinecone_client = Pinecone(api_key=self.pinecone_api_key)
            existing_indexes = [index.name for index in self.pinecone_client.list_indexes()]
            
            if self.pinecone_index_name not in existing_indexes:
                logger.info("Creating Pinecone index: %s", self.pinecone_index_name)
                self.pinecone_client.create_index(
                    name=self.pinecone_index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.pinecone_environment
                    )
                )
            self.index = self.pinecone_client.Index(self.pinecone_index_name)
            logger.info("Pinecone index initialized: %s", self.pinecone_index_name)
        except Exception as e:
            logger.error("Error initializing Pinecone index: %s", str(e))
            raise

    def _initialize_embeddings(self):
        """Initialize OpenAI embeddings"""
        try:
            self.embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
            logger.info("OpenAI Embedding model initialized.")
        except Exception as e:
            logger.error("Error initializing embedding model: %s", str(e))
            raise

    def _initialize_vector_store(self):
        """Initialize Langchain PineconeVectorStore"""
        try:
            self.vector_store = PineconeVectorStore(
                embedding=self.embedding_model,
                index=self.index,
                text_key="text"
            )
            logger.info("Vector store initialized successfully.")
        except Exception as e:
            logger.error("Error initializing vector store: %s", str(e))
            raise

    async def save(self, key: str, metadata: Dict[str, Any], text: str) -> None:
        """
        Save data to vector store
        
        Args:
            key (str): Unique identifier for the data
            metadata (Dict[str, Any]): Additional metadata to store
            text (str): Text content to embed and store
        """
        try:
            logger.info("Saving data to vector store for key: %s", key)
            await self.vector_store.aadd_texts(
                texts=[text],
                metadatas=[{"id": key, **metadata}],
                ids=[key]
            )
            logger.info("Data saved successfully for key: %s", key)
        except Exception as e:
            logger.error("Error saving data for key '%s': %s", key, str(e))
            raise

    async def fetch(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Fetch data by key
        
        Args:
            key (str): Key to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Retrieved data or None if not found
        """
        try:
            logger.info("Fetching data from vector store for key: %s", key)
            results = await self.vector_store.asimilarity_search_with_score(
                query="",
                k=1,
                filter={"id": key}
            )
            
            if not results:
                logger.warning("No data found in vector store for key: %s", key)
                return None
                
            doc, score = results[0]
            return {
                "document": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
        except Exception as e:
            logger.error("Error fetching data for key '%s': %s", key, str(e))
            raise

    async def update(self, key: str, metadata: Dict[str, Any], text: str) -> None:
        """
        Update data in vector store
        
        Args:
            key (str): Key to update
            metadata (Dict[str, Any]): New metadata
            text (str): New text content
        """
        try:
            logger.info("Updating data in vector store for key: %s", key)
            existing_data = await self.fetch(key)
            if not existing_data:
                logger.warning("No existing data found for key: %s. Creating new entry.", key)
            
            await self.vector_store.adelete([key])
            
            await self.save(key, metadata, text)
            logger.info("Data updated successfully for key: %s", key)
        except Exception as e:
            logger.error("Error updating data for key '%s': %s", key, str(e))
            raise

    async def delete(self, key: str) -> bool:
        """
        Delete data from vector store
        
        Args:
            key (str): Key to delete
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            logger.info("Deleting data from vector store for key: %s", key)
            await self.vector_store.adelete([key])
            logger.info("Data deleted successfully for key: %s", key)
            return True
        except Exception as e:
            logger.error("Error deleting data for key '%s': %s", key, str(e))
            return False

    async def natural_query(
        self, 
        query: str, 
        limit: int = 5, 
        filter: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Perform natural language search using vector similarity
        
        Args:
            query (str): Natural language query
            limit (int): Maximum number of results
            filter (Dict[str, Any], optional): Metadata filter conditions
            min_score (float): Minimum similarity score threshold (0.0 to 1.0)
            
        Returns:
            List[Dict[str, Any]]: List of matching documents with scores
        """
        try:
            logger.info(f"Performing natural language query: '{query}'")
            results = await self.vector_store.asimilarity_search_with_score(
                query=query,
                k=limit,
                filter=filter
            )
            
            formatted_results = [
                {
                    "document": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                }
                for doc, score in results
                if score >= min_score
            ]
            
            logger.info(f"Natural query found {len(formatted_results)} matching items")
            return formatted_results

        except Exception as e:
            logger.error(f"Error performing natural query: {e}")
            raise

VectorService = None

def get_vector_service():
    global VectorService
    if not VectorService:
        VectorService = VectorStoreService()
    return VectorService