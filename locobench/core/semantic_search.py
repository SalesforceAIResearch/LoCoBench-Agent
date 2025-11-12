"""
Semantic Search and RAG System for LoCoBench-Agent

Provides vector-based semantic search over codebases, enabling agents to
retrieve relevant code snippets similar to Cursor's @codebase feature.
"""

import logging
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import hashlib

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("numpy not available, semantic search will use fallback mode")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available, will use OpenAI embeddings")


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata"""
    chunk_id: str
    file_path: str
    content: str
    start_line: int
    end_line: int
    chunk_type: str  # 'function', 'class', 'module', 'block'
    language: str
    symbols: List[str] = field(default_factory=list)  # function/class names
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Represents a search result with relevance score"""
    chunk: CodeChunk
    score: float
    rank: int


class CodebaseIndex:
    """
    Vector-based index for semantic code search
    
    Similar to Cursor's @codebase feature, enables semantic search
    across entire codebases using embedding-based retrieval.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        use_openai: bool = False
    ):
        """
        Initialize codebase index
        
        Args:
            model_name: Embedding model to use (sentence-transformers or OpenAI)
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Overlap between chunks for context preservation
            use_openai: Whether to use OpenAI embeddings (requires API key)
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_openai = use_openai
        
        self.chunks: List[CodeChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index_metadata: Dict[str, Any] = {}
        
        # Initialize embedding model
        if use_openai:
            self.embedding_model = None  # Will use OpenAI API
            logger.info("Using OpenAI embeddings for semantic search")
        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(model_name)
                logger.info(f"Loaded sentence-transformers model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load sentence-transformers: {e}, falling back to keyword search")
                self.embedding_model = None
        else:
            self.embedding_model = None
            logger.warning("No embedding model available, using keyword-based fallback")
    
    def index_codebase(
        self,
        project_files: Dict[str, str],
        project_name: str = "unknown"
    ) -> None:
        """
        Index a codebase for semantic search
        
        Args:
            project_files: Dictionary mapping file paths to file contents
            project_name: Name of the project being indexed
        """
        logger.info(f"Indexing codebase: {project_name} ({len(project_files)} files)")
        
        self.chunks = []
        self.index_metadata = {
            "project_name": project_name,
            "num_files": len(project_files),
            "indexed_at": str(Path.cwd())
        }
        
        # Chunk each file
        for file_path, content in project_files.items():
            file_chunks = self._chunk_file(file_path, content)
            self.chunks.extend(file_chunks)
        
        logger.info(f"Created {len(self.chunks)} chunks from {len(project_files)} files")
        
        # Generate embeddings
        if self.embedding_model is not None or self.use_openai:
            self._generate_embeddings()
        else:
            logger.warning("Skipping embedding generation (no model available)")
    
    def _chunk_file(self, file_path: str, content: str) -> List[CodeChunk]:
        """
        Split file into semantic chunks
        
        Attempts to chunk by functions/classes, falls back to line-based chunking
        """
        chunks = []
        language = self._detect_language(file_path)
        
        # Try to chunk by code structure (functions, classes)
        structured_chunks = self._chunk_by_structure(file_path, content, language)
        if structured_chunks:
            return structured_chunks
        
        # Fallback: chunk by lines
        lines = content.split('\n')
        current_chunk = []
        current_start = 1
        
        for i, line in enumerate(lines, 1):
            current_chunk.append(line)
            
            # Create chunk when reaching size limit
            if len(current_chunk) >= self.chunk_size:
                chunk_content = '\n'.join(current_chunk)
                chunk_id = self._generate_chunk_id(file_path, current_start, i)
                
                chunks.append(CodeChunk(
                    chunk_id=chunk_id,
                    file_path=file_path,
                    content=chunk_content,
                    start_line=current_start,
                    end_line=i,
                    chunk_type='block',
                    language=language
                ))
                
                # Keep overlap for context
                overlap_lines = min(self.chunk_overlap, len(current_chunk))
                current_chunk = current_chunk[-overlap_lines:]
                current_start = i - overlap_lines + 1
        
        # Add final chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunk_id = self._generate_chunk_id(file_path, current_start, len(lines))
            chunks.append(CodeChunk(
                chunk_id=chunk_id,
                file_path=file_path,
                content=chunk_content,
                start_line=current_start,
                end_line=len(lines),
                chunk_type='block',
                language=language
            ))
        
        return chunks
    
    def _chunk_by_structure(
        self,
        file_path: str,
        content: str,
        language: str
    ) -> List[CodeChunk]:
        """
        Chunk code by structure (functions, classes)
        
        This is a simplified version - could be enhanced with AST parsing
        """
        chunks = []
        
        # Simple heuristic-based chunking for common patterns
        patterns = {
            'python': ['def ', 'class ', 'async def '],
            'javascript': ['function ', 'class ', 'const ', 'let ', 'var '],
            'typescript': ['function ', 'class ', 'interface ', 'type ', 'const ', 'export '],
            'java': ['public ', 'private ', 'protected ', 'class ', 'interface '],
            'cpp': ['class ', 'struct ', 'void ', 'int ', 'bool ', 'namespace '],
            'c': ['void ', 'int ', 'struct ', 'typedef '],
        }
        
        if language not in patterns:
            return []  # Fallback to line-based chunking
        
        lines = content.split('\n')
        current_chunk_lines = []
        current_start = 1
        current_symbol = None
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Check if line starts a new definition
            is_definition = any(stripped.startswith(pattern) for pattern in patterns[language])
            
            if is_definition and current_chunk_lines:
                # Save previous chunk
                chunk_content = '\n'.join(current_chunk_lines)
                chunk_id = self._generate_chunk_id(file_path, current_start, i - 1)
                
                chunks.append(CodeChunk(
                    chunk_id=chunk_id,
                    file_path=file_path,
                    content=chunk_content,
                    start_line=current_start,
                    end_line=i - 1,
                    chunk_type='function' if 'def ' in stripped or 'function' in stripped else 'class',
                    language=language,
                    symbols=[current_symbol] if current_symbol else []
                ))
                
                # Start new chunk
                current_chunk_lines = [line]
                current_start = i
                current_symbol = self._extract_symbol_name(stripped, language)
            else:
                current_chunk_lines.append(line)
        
        # Add final chunk
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            chunk_id = self._generate_chunk_id(file_path, current_start, len(lines))
            chunks.append(CodeChunk(
                chunk_id=chunk_id,
                file_path=file_path,
                content=chunk_content,
                start_line=current_start,
                end_line=len(lines),
                chunk_type='function' if current_symbol else 'block',
                language=language,
                symbols=[current_symbol] if current_symbol else []
            ))
        
        return chunks
    
    def _generate_embeddings(self) -> None:
        """Generate embeddings for all chunks"""
        if not self.chunks:
            logger.warning("No chunks to embed")
            return
        
        logger.info(f"Generating embeddings for {len(self.chunks)} chunks...")
        
        # Prepare texts for embedding
        texts = [self._prepare_chunk_for_embedding(chunk) for chunk in self.chunks]
        
        if self.use_openai:
            embeddings = self._get_openai_embeddings(texts)
        else:
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
        
        # Store embeddings
        if NUMPY_AVAILABLE:
            self.embeddings = np.array(embeddings)
        
        # Attach embeddings to chunks
        for chunk, embedding in zip(self.chunks, embeddings):
            chunk.embedding = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        
        logger.info(f"Generated embeddings shape: {self.embeddings.shape if self.embeddings is not None else 'N/A'}")
    
    def _prepare_chunk_for_embedding(self, chunk: CodeChunk) -> str:
        """
        Prepare chunk text for embedding
        
        Adds metadata to improve semantic search quality
        """
        # Include file path and symbols for better context
        context_parts = [f"File: {chunk.file_path}"]
        
        if chunk.symbols:
            context_parts.append(f"Symbols: {', '.join(chunk.symbols)}")
        
        context_parts.append(f"Code:\n{chunk.content}")
        
        return "\n".join(context_parts)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_language: Optional[str] = None,
        filter_file_pattern: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Semantic search for relevant code chunks
        
        Args:
            query: Natural language query (e.g., "user authentication logic")
            top_k: Number of top results to return
            filter_language: Optional language filter (e.g., 'python')
            filter_file_pattern: Optional file pattern filter (e.g., 'src/')
        
        Returns:
            List of SearchResult objects ranked by relevance
        """
        if not self.chunks:
            logger.warning("No chunks indexed, returning empty results")
            return []
        
        # Filter chunks if requested
        candidate_chunks = self.chunks
        if filter_language:
            candidate_chunks = [c for c in candidate_chunks if c.language == filter_language]
        if filter_file_pattern:
            candidate_chunks = [c for c in candidate_chunks if filter_file_pattern in c.file_path]
        
        if not candidate_chunks:
            logger.warning(f"No chunks match filters (language={filter_language}, pattern={filter_file_pattern})")
            return []
        
        # Semantic search if embeddings available
        if self.embeddings is not None and (self.embedding_model is not None or self.use_openai):
            return self._semantic_search(query, candidate_chunks, top_k)
        else:
            # Fallback to keyword search
            return self._keyword_search(query, candidate_chunks, top_k)
    
    def _semantic_search(
        self,
        query: str,
        candidate_chunks: List[CodeChunk],
        top_k: int
    ) -> List[SearchResult]:
        """Semantic search using embeddings"""
        # Generate query embedding
        if self.use_openai:
            query_embedding = self._get_openai_embeddings([query])[0]
        else:
            query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate similarity scores
        scores = []
        for chunk in candidate_chunks:
            if chunk.embedding is None:
                continue
            
            # Cosine similarity
            chunk_emb = np.array(chunk.embedding)
            query_emb = np.array(query_embedding)
            
            similarity = np.dot(query_emb, chunk_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb)
            )
            scores.append((chunk, float(similarity)))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (chunk, score) in enumerate(scores[:top_k], 1):
            results.append(SearchResult(chunk=chunk, score=score, rank=rank))
        
        return results
    
    def _keyword_search(
        self,
        query: str,
        candidate_chunks: List[CodeChunk],
        top_k: int
    ) -> List[SearchResult]:
        """Fallback keyword-based search"""
        query_tokens = set(query.lower().split())
        
        scores = []
        for chunk in candidate_chunks:
            content_lower = chunk.content.lower()
            file_path_lower = chunk.file_path.lower()
            
            # Count matching tokens
            matches = sum(1 for token in query_tokens if token in content_lower or token in file_path_lower)
            
            # Boost if symbols match
            symbol_matches = sum(1 for symbol in chunk.symbols if symbol.lower() in query.lower())
            
            score = matches + (symbol_matches * 2)  # Symbols weighted higher
            
            if score > 0:
                scores.append((chunk, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (chunk, score) in enumerate(scores[:top_k], 1):
            # Normalize score to 0-1 range
            normalized_score = score / (len(query_tokens) + len(chunk.symbols))
            results.append(SearchResult(chunk=chunk, score=normalized_score, rank=rank))
        
        return results
    
    def _get_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from OpenAI API (supports gateway)"""
        try:
            import openai
            import os
            
            # Support for Salesforce Research OpenAI Gateway
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            
            client_kwargs = {}
            if base_url:
                # Gateway requires X-Api-Key header and dummy api_key
                client_kwargs["base_url"] = base_url
                client_kwargs["api_key"] = "dummy"
                client_kwargs["default_headers"] = {"X-Api-Key": api_key}
                logger.info(f"Using OpenAI Gateway for embeddings: {base_url}")
            else:
                # Direct OpenAI API
                client_kwargs["api_key"] = api_key
            
            client = openai.OpenAI(**client_kwargs)
            
            # Batch texts for efficiency
            embeddings = []
            batch_size = 100
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                embeddings.extend([item.embedding for item in response.data])
            
            return embeddings
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise
    
    def save_index(self, output_path: Path) -> None:
        """Save index to disk for reuse"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        index_data = {
            "chunks": [self._chunk_to_dict(c) for c in self.chunks],
            "metadata": self.index_metadata,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None,
            "config": {
                "model_name": self.model_name,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        logger.info(f"Saved index to {output_path}")
    
    def load_index(self, input_path: Path) -> None:
        """Load index from disk"""
        with open(input_path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.chunks = [self._dict_to_chunk(d) for d in index_data["chunks"]]
        self.index_metadata = index_data["metadata"]
        
        if index_data["embeddings"] and NUMPY_AVAILABLE:
            self.embeddings = np.array(index_data["embeddings"])
        
        logger.info(f"Loaded index from {input_path} ({len(self.chunks)} chunks)")
    
    # Helper methods
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = Path(file_path).suffix.lower()
        lang_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.rs': 'rust',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.cs': 'csharp',
        }
        return lang_map.get(ext, 'unknown')
    
    def _generate_chunk_id(self, file_path: str, start_line: int, end_line: int) -> str:
        """Generate unique chunk ID"""
        content = f"{file_path}:{start_line}-{end_line}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _extract_symbol_name(self, line: str, language: str) -> Optional[str]:
        """Extract function/class name from definition line"""
        import re
        
        patterns = {
            'python': [r'def\s+(\w+)', r'class\s+(\w+)'],
            'javascript': [r'function\s+(\w+)', r'class\s+(\w+)', r'const\s+(\w+)\s*='],
            'typescript': [r'function\s+(\w+)', r'class\s+(\w+)', r'interface\s+(\w+)', r'const\s+(\w+)\s*='],
            'java': [r'class\s+(\w+)', r'interface\s+(\w+)', r'\w+\s+(\w+)\s*\('],
        }
        
        if language not in patterns:
            return None
        
        for pattern in patterns[language]:
            match = re.search(pattern, line)
            if match:
                return match.group(1)
        
        return None
    
    def _chunk_to_dict(self, chunk: CodeChunk) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization"""
        return {
            "chunk_id": chunk.chunk_id,
            "file_path": chunk.file_path,
            "content": chunk.content,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "chunk_type": chunk.chunk_type,
            "language": chunk.language,
            "symbols": chunk.symbols,
            "embedding": chunk.embedding,
            "metadata": chunk.metadata
        }
    
    def _dict_to_chunk(self, data: Dict[str, Any]) -> CodeChunk:
        """Convert dictionary to CodeChunk"""
        return CodeChunk(**data)


class SemanticCodeRetriever:
    """
    High-level interface for semantic code retrieval
    
    Manages codebase indexing and provides search capabilities
    similar to Cursor's @codebase feature.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, use_openai: bool = False):
        """
        Initialize semantic retriever
        
        Args:
            cache_dir: Directory to cache indices
            use_openai: Whether to use OpenAI embeddings (requires API key)
        """
        self.cache_dir = cache_dir or Path.home() / ".locobench" / "semantic_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_openai = use_openai
        
        self.current_index: Optional[CodebaseIndex] = None
        self.current_project: Optional[str] = None
    
    def index_project(
        self,
        project_files: Dict[str, str],
        project_name: str,
        force_reindex: bool = False
    ) -> CodebaseIndex:
        """
        Index a project for semantic search
        
        Args:
            project_files: Dictionary mapping file paths to contents
            project_name: Project identifier
            force_reindex: Force reindexing even if cache exists
        
        Returns:
            CodebaseIndex object
        """
        cache_path = self.cache_dir / f"{project_name}.index"
        
        # Check cache
        if cache_path.exists() and not force_reindex:
            logger.info(f"Loading cached index for {project_name}")
            index = CodebaseIndex()
            index.load_index(cache_path)
            self.current_index = index
            self.current_project = project_name
            return index
        
        # Create new index
        logger.info(f"Creating new index for {project_name}")
        index = CodebaseIndex(use_openai=self.use_openai)
        index.index_codebase(project_files, project_name)
        
        # Save to cache
        index.save_index(cache_path)
        
        self.current_index = index
        self.current_project = project_name
        return index
    
    def search_codebase(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search current codebase semantically
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            **kwargs: Additional filters (language, file_pattern)
        
        Returns:
            List of SearchResult objects
        """
        if self.current_index is None:
            raise ValueError("No codebase indexed. Call index_project() first.")
        
        return self.current_index.search(query, top_k=top_k, **kwargs)
    
    def cleanup_current_index(self) -> None:
        """
        Delete the cached index file for the current project to free disk space
        
        This should be called after scenario evaluation completes.
        The index can be regenerated if needed in future evaluations.
        """
        if self.current_project is None:
            return
        
        cache_path = self.cache_dir / f"{self.current_project}.index"
        
        if cache_path.exists():
            try:
                cache_path.unlink()
                logger.debug(f"✅ Deleted semantic index: {cache_path.name}")
            except Exception as e:
                logger.warning(f"Failed to delete semantic index {cache_path.name}: {e}")
        
        # Clear current references
        self.current_index = None
        self.current_project = None

