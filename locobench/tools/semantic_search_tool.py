"""
Semantic Search Tool for LoCoBench-Agent

Provides semantic code search capabilities to agents, similar to Cursor's @codebase feature.
"""

import logging
from typing import Dict, Any, Optional
from ..core.tool_registry import Tool, ToolCategory

logger = logging.getLogger(__name__)


class ToolResult:
    """Result of a tool execution"""
    def __init__(self, success: bool, output: str, error: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        self.success = success
        self.output = output
        self.error = error
        self.metadata = metadata or {}


class SemanticSearchTool(Tool):
    """
    Semantic code search tool
    
    Allows agents to search the codebase semantically using natural language queries.
    Similar to Cursor's @codebase feature.
    """
    
    def __init__(self, semantic_retriever=None):
        """
        Initialize semantic search tool
        
        Args:
            semantic_retriever: SemanticCodeRetriever instance (optional, will be set by session)
        """
        super().__init__(
            name="semantic_search",
            description=(
                "Search the codebase semantically using natural language. "
                "Use this to find relevant code when you're not sure of exact file names. "
                "Example: 'user authentication logic' will find all authentication-related code."
            ),
            category=ToolCategory.SEARCH
        )
        self.semantic_retriever = semantic_retriever
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for function calling"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query describing what code you're looking for (e.g., 'user authentication logic', 'database connection code')"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5, max: 10)",
                        "default": 5
                    },
                    "language": {
                        "type": "string",
                        "description": "Optional: filter by programming language (e.g., 'python', 'javascript', 'java')"
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "Optional: filter by file path pattern (e.g., 'src/', 'test/', '.py')"
                    }
                },
                "required": ["query"]
            }
        }
    
    async def execute(
        self,
        query: str,
        top_k: int = 5,
        language: Optional[str] = None,
        file_pattern: Optional[str] = None
    ) -> ToolResult:
        """
        Execute semantic search
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            language: Optional language filter
            file_pattern: Optional file pattern filter
        
        Returns:
            ToolResult with search results
        """
        try:
            if self.semantic_retriever is None:
                return ToolResult(
                    success=False,
                    output="Semantic search not available (retriever not initialized)",
                    error="Semantic retriever not set"
                )
            
            # Validate top_k
            top_k = max(1, min(top_k, 10))  # Limit to 1-10
            
            logger.info(f"Semantic search: query='{query}', top_k={top_k}, language={language}, pattern={file_pattern}")
            
            # Perform search
            results = self.semantic_retriever.search_codebase(
                query=query,
                top_k=top_k,
                filter_language=language,
                filter_file_pattern=file_pattern
            )
            
            if not results:
                return ToolResult(
                    success=True,
                    output=f"No results found for query: '{query}'\n\nTry:\n- Broader query terms\n- Different keywords\n- Removing filters"
                )
            
            # Format results
            output_parts = [
                f"Found {len(results)} relevant code snippets for: '{query}'\n",
                "=" * 80
            ]
            
            for result in results:
                chunk = result.chunk
                output_parts.append(f"\n📄 Result #{result.rank} - {chunk.file_path}")
                output_parts.append(f"   Relevance: {result.score:.2%}")
                output_parts.append(f"   Lines: {chunk.start_line}-{chunk.end_line}")
                output_parts.append(f"   Type: {chunk.chunk_type}")
                
                if chunk.symbols:
                    output_parts.append(f"   Symbols: {', '.join(chunk.symbols)}")
                
                output_parts.append(f"\n```{chunk.language}")
                
                # Show code snippet (limit to 500 chars for readability)
                content = chunk.content
                if len(content) > 500:
                    content = content[:500] + f"\n... (truncated, {len(chunk.content) - 500} more chars)"
                
                output_parts.append(content)
                output_parts.append("```\n")
            
            output_parts.append("=" * 80)
            output_parts.append(f"\nTip: Use file_system_read_file to read full files")
            
            return ToolResult(
                success=True,
                output="\n".join(output_parts),
                metadata={
                    "num_results": len(results),
                    "query": query,
                    "files_found": [r.chunk.file_path for r in results]
                }
            )
        
        except Exception as e:
            logger.error(f"Semantic search error: {e}", exc_info=True)
            return ToolResult(
                success=False,
                output=f"Semantic search failed: {str(e)}",
                error=str(e)
            )

