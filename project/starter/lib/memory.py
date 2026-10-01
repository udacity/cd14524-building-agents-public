from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import copy

from lib.documents import Document, Corpus
from lib.vector_db import VectorStoreManager,QueryResult


class SessionNotFoundError(Exception):
    """Raised when attempting to access a session that doesn't exist"""
    pass


@dataclass
class ShortTermMemory():
    """Manage the history of objects across multiple sessions"""
    sessions: Dict[str, List[Any]] = field(default_factory=lambda: {})

    def __post_init__(self):
        """Initialize the default session"""
        self.create_session("default")

    def __str__(self) -> str:
        session_ids = list(self.sessions.keys())
        return f"Memory(sessions={session_ids})"

    def __repr__(self) -> str:
        return self.__str__()

    def create_session(self, session_id: str) -> bool:
        """Create a new session
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            bool: True if session was created, False if it already existed
        """
        if session_id in self.sessions:
            return False
        self.sessions[session_id] = []
        return True

    def delete_session(self, session_id: str) -> bool:
        """Delete a session
        
        Args:
            session_id: Session to delete
            
        Returns:
            bool: True if session was deleted, False if it didn't exist
            
        Raises:
            ValueError: If attempting to delete the default session
        """
        if session_id == "default":
            raise ValueError("Cannot delete the default session")
        if session_id not in self.sessions:
            return False
        del self.sessions[session_id]
        return True

    def _validate_session(self, session_id: str):
        """Validate that a session exists
        
        Args:
            session_id: Session ID to validate
            
        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        if session_id not in self.sessions:
            raise SessionNotFoundError(f"Session '{session_id}' not found")

    def add(self, object: Any, session_id: Optional[str] = None):
        """Add a new object to the history
        
        Args:
            object: Object to add to history
            session_id: Optional session ID to add to (uses default if None)
            
        Raises:
            SessionNotFoundError: If specified session doesn't exist
        """
        session_id = session_id or "default"
        self._validate_session(session_id)
        self.sessions[session_id].append(copy.deepcopy(object))

    def get_all_objects(self, session_id: Optional[str] = None) -> List[Any]:
        """Get all objects for a session
        
        Args:
            session_id: Optional session ID (uses default if None)
            
        Returns:
            List of objects in the session
            
        Raises:
            SessionNotFoundError: If specified session doesn't exist
        """
        session_id = session_id or "default"
        self._validate_session(session_id)
        return [copy.deepcopy(obj) for obj in self.sessions[session_id]]

    def get_last_object(self, session_id: Optional[str] = None) -> Optional[Any]:
        """Get the most recent object for a session
        
        Args:
            session_id: Optional session ID (uses default if None)
            
        Returns:
            The last object in the session if it exists, None if session is empty
            
        Raises:
            SessionNotFoundError: If specified session doesn't exist
        """
        objects = self.get_all_objects(session_id)
        return objects[-1] if objects else None

    def get_all_sessions(self) -> List[str]:
        """Get all session IDs"""
        return list(self.sessions.keys())

    def reset(self, session_id: Optional[str] = None):
        """Reset memory for a specific session or all sessions
        
        Args:
            session_id: Optional session ID to reset. If None, resets all sessions.
            
        Raises:
            SessionNotFoundError: If specified session doesn't exist
        """
        if session_id is None:
            # Reset all sessions to empty lists
            for sid in self.sessions:
                self.sessions[sid] = []
        else:
            self._validate_session(session_id)
            self.sessions[session_id] = []

    def pop(self, session_id: Optional[str] = None) -> Optional[Any]:
        """Remove and return the last object from a session
        
        Args:
            session_id: Optional session ID to pop from (uses default if None)
            
        Returns:
            The last object in the session if it exists, None if session is empty
            
        Raises:
            SessionNotFoundError: If specified session doesn't exist
        """
        session_id = session_id or "default"
        self._validate_session(session_id)
        
        if not self.sessions[session_id]:
            return None
        return self.sessions[session_id].pop()

@dataclass
class MemoryFragment:
    """
    Represents a single piece of memory information stored in the long-term memory system.
    
    This class encapsulates user preferences, facts, or contextual information that can be
    retrieved later to provide personalized responses in conversational AI applications.
    
    Attributes:
        content (str): The actual memory content or information to be stored
        owner (str): Identifier for the user who owns this memory fragment
        namespace (str): Logical grouping for organizing related memories (default: "default")
        timestamp (int): Unix timestamp when the memory was created (auto-generated)
    """
    content: str
    owner: str 
    namespace: str = "default"
    timestamp: int = field(default_factory=lambda: int(datetime.now().timestamp()))


@dataclass
class MemorySearchResult:
    """
    Container for the results of a memory search operation.
    
    Encapsulates both the retrieved memory fragments and associated metadata
    such as distance scores from the vector search.
    
    Attributes:
        fragments (List[MemoryFragment]): List of memory fragments matching the search query
        metadata (Dict): Additional information about the search results (e.g., distances, scores)
    """
    fragments: List[MemoryFragment]
    metadata: Dict

@dataclass
class TimestampFilter:
    """
    Filter criteria for time-based memory searches.
    
    Allows filtering memory fragments based on when they were created,
    enabling retrieval of recent memories or memories from specific time periods.
    
    Attributes:
        greater_than_value (int, optional): Unix timestamp - only return memories created after this time
        lower_than_value (int, optional): Unix timestamp - only return memories created before this time
    """
    greater_than_value: int = None
    lower_than_value: int = None

class LongTermMemory:
    """
    Manages persistent memory storage and retrieval using vector embeddings.
    
    This class provides a high-level interface for storing and searching user memories,
    preferences, and contextual information across conversation sessions. It uses
    vector similarity search to find relevant memories based on semantic meaning.
    
    The memory system supports:
    - Multi-user memory isolation
    - Namespace-based organization
    - Time-based filtering
    - Semantic similarity search
    """
    def __init__(self, db: VectorStoreManager, reset: bool = False):
        # reset=True starts from an empty memory. By default, what an earlier session stored is kept
        # (only possible when db was created with persist_path).
        self.vector_store = (
            db.create_store("long_term_memory", force=True) if reset
            else db.get_or_create_store("long_term_memory")
        )

    def get_namespaces(self) -> List[str]:
        """
        Retrieve all unique namespaces currently stored in memory.
        
        Useful for understanding how memories are organized and for
        administrative purposes.
        
        Returns:
            List[str]: List of unique namespace identifiers
        """
        results = self.vector_store.get()
        namespaces = [r["metadatas"][0]["namespace"] for r in results]
        return namespaces

    def register(self, memory_fragment:MemoryFragment, metadata:Optional[Dict[str, str]]=None):
        """
        Store a new memory fragment in the long-term memory system.
        
        The memory is converted to a vector embedding and stored with associated
        metadata for later retrieval. Additional metadata can be provided to
        enhance searchability.
        
        Args:
            memory_fragment (MemoryFragment): The memory content to store
            metadata (Optional[Dict[str, str]]): Additional metadata to associate with the memory
        """
        complete_metadata = {
            "owner": memory_fragment.owner,
            "namespace": memory_fragment.namespace,
            "timestamp": memory_fragment.timestamp,
        }
        if metadata:
            complete_metadata.update(metadata)

        self.vector_store.add(
            Document(
                content=memory_fragment.content,
                metadata=complete_metadata,
            )
        )

    def search(self, query_text:str, owner:str, limit:int=3,
               timestamp_filter:Optional[TimestampFilter]=None, 
               namespace:Optional[str]="default") -> MemorySearchResult:
        """
        Search for relevant memories using semantic similarity.
        
        Performs a vector similarity search to find memories that are semantically
        related to the query text. Results are filtered by owner, namespace, and
        optionally by timestamp range.
        
        Args:
            query_text (str): The search query to find similar memories
            owner (str): User identifier to filter memories by ownership
            limit (int): Maximum number of results to return (default: 3)
            timestamp_filter (Optional[TimestampFilter]): Time-based filtering criteria
            namespace (Optional[str]): Namespace to search within (default: "default")
            
        Returns:
            MemorySearchResult: Container with matching memory fragments and metadata
        """

        where = {
            "$and": [
                {
                    "namespace": {
                        "$eq": namespace
                    }
                },
                {
                    "owner": {
                        "$eq": owner
                    }
                },
            ]
        }

        if timestamp_filter:
            if timestamp_filter.greater_than_value:
                where["$and"].append({
                    "timestamp": {
                        "$gt": timestamp_filter.greater_than_value,
                    }
                })
            if timestamp_filter.lower_than_value:
                where["$and"].append({
                    "timestamp": {
                        "$lt": timestamp_filter.lower_than_value,
                    }
                })

        result:QueryResult = self.vector_store.query(
            query_texts=[query_text],
            n_results=limit,
            where=where
        )

        fragments = []
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        for content, meta in zip(documents, metadatas):
            owner = meta.get("owner")
            namespace = meta.get("namespace", "default")
            timestamp = meta.get("timestamp")

            fragment = MemoryFragment(
                content=content,
                owner=owner,
                namespace=namespace,
                timestamp=timestamp
            )

            fragments.append(fragment)
        
        result_metadata = {
            "distances": result.get("distances", [[]])[0]
        }

        return MemorySearchResult(
            fragments=fragments,
            metadata=result_metadata
        )
