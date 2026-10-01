from typing import List, Optional, Dict, Any, Union
from typing_extensions import TypedDict
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.models.Collection import Collection as ChromaCollection
from chromadb.api.types import EmbeddingFunction, QueryResult, GetResult

from lib.loaders import PDFLoader
from lib.documents import Document, Corpus


class VectorStore:
    """
    High-level interface for vector database operations using ChromaDB.
    
    This class provides a simplified API for storing and querying document embeddings
    in a ChromaDB collection. It handles the conversion between our Document/Corpus
    abstractions and ChromaDB's expected data formats, making vector operations
    more intuitive and type-safe.
    
    The VectorStore supports:
    - Adding individual documents, document lists, or corpus collections
    - Semantic similarity search with filtering capabilities
    - Metadata-based document retrieval
    - Automatic embedding generation via OpenAI
    """

    def __init__(self, chroma_collection: ChromaCollection):
        self._collection = chroma_collection

    def add(self, item: Union[Document, Corpus, List[Document]]):
        """
        Add documents to the vector store with automatic embedding generation.
        
        This method accepts various input formats and normalizes them to the
        ChromaDB batch format. Documents are automatically embedded using the
        collection's configured embedding function (typically OpenAI).
        
        Args:
            item (Union[Document, Corpus, List[Document]]): Documents to add.
                Can be a single Document, a Corpus collection, or a list of Documents.
                
        Raises:
            TypeError: If the input type is not supported or if a list contains
                non-Document objects.
                
        Example:
            >>> store.add(Document(content="AI is transforming healthcare"))
            >>> store.add([doc1, doc2, doc3])  # Batch add
            >>> store.add(Corpus([doc1, doc2]))  # Add corpus
        """
        if isinstance(item, Document):
            item = Corpus([item])
        elif isinstance(item, list):
            if not all(isinstance(doc, Document) for doc in item):
                raise TypeError("List must contain Document objects only.")
            item = Corpus(item)
        elif not isinstance(item, Corpus):
            raise TypeError("item must be Document, Corpus, or List[Document].")

        item_dict = item.to_dict()

        self._collection.add(
            documents=item_dict["contents"],
            ids=item_dict["ids"],
            metadatas=item_dict["metadatas"]
        )

    def query(self, query_texts: str | List[str], n_results: int = 3,
              where: Optional[Dict[str, Any]] = None,
              where_document: Optional[Dict[str, Any]] = None) -> QueryResult:
        """
        Perform semantic similarity search against stored documents.
        
        This method finds documents that are semantically similar to the query
        text using vector embeddings. Results are ranked by cosine similarity
        and can be filtered using metadata or document content conditions.
        
        Args:
            query_texts (List[str]): List of query strings to search for
            n_results (int): Maximum number of results to return per query (default: 3)
            where (Optional[Dict[str, Any]]): Metadata filter conditions using
                ChromaDB query syntax (e.g., {"author": "Smith"})
            where_document (Optional[Dict[str, Any]]): Document content filter
                conditions using ChromaDB query syntax
                
        Returns:
            QueryResult: ChromaDB query result containing documents, distances,
                metadata, and IDs for the most similar documents
                
        Example:
            >>> results = store.query(
            ...     query_texts=["machine learning algorithms"],
            ...     n_results=5,
            ...     where={"category": "technical"}
            ... )
            >>> for doc, distance in zip(results['documents'][0], results['distances'][0]):
            ...     print(f"Similarity: {1-distance:.3f}, Content: {doc[:100]}...")
        """
        return self._collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=['documents', 'distances', 'metadatas']
        )

    def get(self, ids: Optional[List[str]] = None, 
            where: Optional[Dict[str, Any]] = None,
            limit: Optional[int] = None) -> GetResult:
        """
        Retrieve documents by ID or metadata filters without similarity search.
        
        This method performs direct document retrieval based on exact ID matches
        or metadata filter conditions, without computing embedding similarities.
        Useful for fetching specific documents or browsing collections.
        
        Args:
            ids (Optional[List[str]]): Specific document IDs to retrieve
            where (Optional[Dict[str, Any]]): Metadata filter conditions
            limit (Optional[int]): Maximum number of documents to return
            
        Returns:
            GetResult: ChromaDB result containing the requested documents
                with their metadata and IDs
                
        Example:
            >>> # Get specific documents by ID
            >>> docs = store.get(ids=["doc-123", "doc-456"])
            >>> 
            >>> # Get all documents from a specific source
            >>> docs = store.get(where={"source": "research_papers"}, limit=10)
        """
        return self._collection.get(
            ids=ids,
            where=where,
            limit=limit,
            include=['documents', 'distances', 'metadatas']
        )

class VectorStoreManager:
    """
    Factory and lifecycle manager for ChromaDB vector stores.
    
    This class handles the creation, configuration, and management of ChromaDB
    collections with OpenAI embeddings. It provides a centralized way to manage
    multiple vector stores within an application, handling the underlying ChromaDB
    client and embedding function configuration.
    
    Key responsibilities:
    - ChromaDB client initialization and management
    - OpenAI embedding function configuration
    - Vector store creation with consistent settings
    - Store lifecycle management (create, get, delete)
    """

    def __init__(self, openai_api_key: str, persist_path: Optional[str] = None):
        # In-memory by default; pass persist_path to keep stores on disk between sessions.
        self.chroma_client = (
            chromadb.PersistentClient(path=persist_path) if persist_path else chromadb.Client()
        )
        self.embedding_function = self._create_embedding_function(openai_api_key)

    def _create_embedding_function(self, api_key: str) -> EmbeddingFunction:
        embeddings_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key
        )
        return embeddings_fn

    def __repr__(self):
        return f"VectorStoreManager():{self.chroma_client}"

    def get_store(self, name: str) -> Optional[VectorStore]:
        try:
            chroma_collection = self.chroma_client.get_collection(name)
            return VectorStore(chroma_collection)
        except Exception:
            return None

    def create_store(self, store_name: str, force: bool = False) -> VectorStore:
        if force:
            self.delete_store(store_name)

        try:
            chroma_collection = self.chroma_client.create_collection(
                name=store_name,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            print(f"Pass `force=True` or use `get_or_create_store` method")

        return VectorStore(chroma_collection)

    def get_or_create_store(self, store_name: str) -> VectorStore:
        chroma_collection = self.chroma_client.get_or_create_collection(
            name=store_name,
            embedding_function=self.embedding_function
        )
        return VectorStore(chroma_collection)

    def delete_store(self, store_name: str):
        try:
            self.chroma_client.delete_collection(name=store_name)
        except Exception:
            pass  # Store doesn't exist yet


class CorpusLoaderService:
    """
    Service for loading documents from various sources into vector stores.
    
    This class provides convenient methods for loading and processing documents
    from different file formats (currently PDF) into vector stores. It handles
    the entire pipeline from file loading to vector store insertion.
    
    The service abstracts away the complexity of:
    - Document loading and parsing
    - Vector store creation and management
    - Batch document insertion
    - Progress reporting and error handling
    """

    def __init__(self, vector_store_manager: VectorStoreManager):
        self.manager = vector_store_manager

    def load_pdf(self, store_name: str, pdf_path: str) -> VectorStore:
        """
        Load a PDF file into a vector store.
        
        This method handles the complete pipeline of loading a PDF document,
        parsing its content into pages/chunks, and storing them in a vector
        store with embeddings. Each page becomes a separate document in the store.
        
        Args:
            store_name (str): Name of the vector store to create or use
            pdf_path (str): Path to the PDF file to load
            
        Returns:
            VectorStore: The vector store containing the loaded PDF content
            
        Example:
            >>> loader = CorpusLoaderService(manager)
            >>> store = loader.load_pdf("research_papers", "paper.pdf")
            >>> # PDF is now searchable in the vector store
            >>> results = store.query(["machine learning methodology"])
        """
        store = self.manager.get_or_create_store(store_name)
        print(f"VectorStore `{store_name}` ready!")

        loader = PDFLoader(pdf_path)
        document = loader.load()
        store.add(document)
        print(f"Pages from `{pdf_path}` added!")

        return store
