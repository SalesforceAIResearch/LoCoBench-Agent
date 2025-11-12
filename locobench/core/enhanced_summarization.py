"""
Enhanced Conversation Summarization using LLM

Provides intelligent conversation summarization similar to production
coding assistants like Cursor and GitHub Copilot.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SummaryType(Enum):
    """Types of summaries that can be generated"""
    CONVERSATION_HISTORY = "conversation_history"
    CODE_CHANGES = "code_changes"
    DECISIONS_MADE = "decisions_made"
    ERRORS_ENCOUNTERED = "errors_encountered"
    ARCHITECTURAL_OVERVIEW = "architectural_overview"


@dataclass
class ConversationSummary:
    """Structured summary of conversation"""
    summary_type: SummaryType
    content: str
    turns_covered: List[int]
    key_points: List[str]
    important_context: Dict[str, Any]
    token_count: int


class LLMSummarizer:
    """
    LLM-based conversation summarizer
    
    Uses the same LLM that the agent uses to generate intelligent
    summaries of conversation history, preserving critical information
    while dramatically reducing token usage.
    """
    
    def __init__(self, llm_client: Any, model_name: str = "gpt-4o-mini"):
        """
        Initialize summarizer
        
        Args:
            llm_client: LLM client (OpenAI, Anthropic, etc.)
            model_name: Model to use for summarization (typically a smaller/cheaper model)
        """
        self.llm_client = llm_client
        self.model_name = model_name
    
    async def summarize_conversation(
        self,
        conversation_turns: List[Dict[str, Any]],
        focus: Optional[SummaryType] = None
    ) -> ConversationSummary:
        """
        Summarize conversation turns using LLM
        
        Args:
            conversation_turns: List of conversation turns to summarize
            focus: Optional focus for the summary (e.g., code_changes, decisions)
        
        Returns:
            ConversationSummary object
        """
        if not conversation_turns:
            return ConversationSummary(
                summary_type=focus or SummaryType.CONVERSATION_HISTORY,
                content="No conversation to summarize.",
                turns_covered=[],
                key_points=[],
                important_context={},
                token_count=0
            )
        
        # Build prompt for summarization
        prompt = self._build_summarization_prompt(conversation_turns, focus)
        
        # Get summary from LLM
        try:
            response = await self._call_llm(prompt)
            summary_content = response.get("content", "")
            
            # Extract key points and context
            key_points = self._extract_key_points(summary_content)
            important_context = self._extract_important_context(conversation_turns, summary_content)
            
            return ConversationSummary(
                summary_type=focus or SummaryType.CONVERSATION_HISTORY,
                content=summary_content,
                turns_covered=[t.get("turn_number", i) for i, t in enumerate(conversation_turns, 1)],
                key_points=key_points,
                important_context=important_context,
                token_count=len(summary_content.split())  # Rough estimate
            )
        
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}, falling back to extractive summary")
            return self._fallback_extractive_summary(conversation_turns, focus)
    
    def _build_summarization_prompt(
        self,
        conversation_turns: List[Dict[str, Any]],
        focus: Optional[SummaryType]
    ) -> str:
        """Build prompt for LLM summarization"""
        
        # Base instructions
        instructions = """You are summarizing a conversation between a coding agent and a user/system. 
Your goal is to create a concise but information-dense summary that preserves ALL critical details needed for the agent to continue working effectively.

CRITICAL INFORMATION TO PRESERVE:
- Variable names, function names, class names mentioned
- File paths and locations
- Specific error messages and their resolutions
- Design decisions and their rationale
- Task progress and next steps
- Tool calls made and their results
- Any constraints or requirements specified

WHAT TO OMIT:
- Verbose explanations
- Repeated information
- Small talk or acknowledgments
- Detailed code that can be retrieved later

Format your summary as:
1. CONTEXT: What was being worked on
2. ACTIONS: What was done (tools used, code written, tests run)
3. OUTCOMES: Results, errors fixed, progress made
4. NEXT STEPS: What needs to be done next
5. IMPORTANT REFERENCES: Key files, functions, variables to remember
"""
        
        # Add focus-specific instructions
        if focus == SummaryType.CODE_CHANGES:
            instructions += "\n\nFOCUS: Emphasize code changes - what was modified, added, or deleted."
        elif focus == SummaryType.DECISIONS_MADE:
            instructions += "\n\nFOCUS: Emphasize design decisions and their rationale."
        elif focus == SummaryType.ERRORS_ENCOUNTERED:
            instructions += "\n\nFOCUS: Emphasize errors encountered and how they were resolved."
        
        # Add conversation turns
        conversation_text = "\n\n---\n\n".join([
            f"Turn {turn.get('turn_number', i)}:\nRole: {turn.get('role', 'unknown')}\nContent: {turn.get('content', '')[:1000]}..."
            for i, turn in enumerate(conversation_turns, 1)
        ])
        
        prompt = f"""{instructions}

CONVERSATION TO SUMMARIZE:

{conversation_text}

SUMMARY:"""
        
        return prompt
    
    async def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call LLM for summarization"""
        # This will be implemented based on the specific LLM client
        # For now, return a placeholder
        
        try:
            # Attempt to use OpenAI client
            if hasattr(self.llm_client, 'chat'):
                response = await self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,  # Lower temperature for factual summarization
                    max_tokens=1000
                )
                return {"content": response.choices[0].message.content}
            
            # Attempt to use Anthropic client
            elif hasattr(self.llm_client, 'messages'):
                response = await self.llm_client.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000
                )
                return {"content": response.content[0].text}
            
            else:
                raise ValueError("Unsupported LLM client")
        
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _extract_key_points(self, summary_content: str) -> List[str]:
        """Extract key points from summary"""
        key_points = []
        
        # Look for bullet points or numbered lists
        lines = summary_content.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('-') or stripped.startswith('•') or stripped.startswith('*'):
                key_points.append(stripped[1:].strip())
            elif stripped and stripped[0].isdigit() and '.' in stripped[:3]:
                key_points.append(stripped.split('.', 1)[1].strip())
        
        return key_points[:10]  # Limit to top 10
    
    def _extract_important_context(
        self,
        conversation_turns: List[Dict[str, Any]],
        summary_content: str
    ) -> Dict[str, Any]:
        """Extract important context from conversation and summary"""
        context = {
            "files_mentioned": set(),
            "functions_mentioned": set(),
            "errors_encountered": [],
            "decisions_made": []
        }
        
        # Scan conversation for important patterns
        for turn in conversation_turns:
            content = turn.get("content", "")
            
            # Extract file paths (simple heuristic)
            import re
            file_patterns = re.findall(r'[\w/]+\.\w{2,4}', content)
            context["files_mentioned"].update(file_patterns)
            
            # Extract function names (simple heuristic)
            function_patterns = re.findall(r'\b\w+\([^)]*\)', content)
            context["functions_mentioned"].update([f.split('(')[0] for f in function_patterns])
            
            # Extract errors (simple heuristic)
            if "error" in content.lower() or "exception" in content.lower():
                error_lines = [line for line in content.split('\n') if 'error' in line.lower() or 'exception' in line.lower()]
                context["errors_encountered"].extend(error_lines[:2])  # Limit per turn
        
        # Convert sets to lists for JSON serialization
        context["files_mentioned"] = list(context["files_mentioned"])[:20]
        context["functions_mentioned"] = list(context["functions_mentioned"])[:20]
        context["errors_encountered"] = context["errors_encountered"][:10]
        
        return context
    
    def _fallback_extractive_summary(
        self,
        conversation_turns: List[Dict[str, Any]],
        focus: Optional[SummaryType]
    ) -> ConversationSummary:
        """Fallback extractive summarization if LLM fails"""
        
        summary_parts = []
        
        for turn in conversation_turns:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            
            # Extract first sentence or first 100 chars
            sentences = content.split('.')
            first_sentence = sentences[0][:200] if sentences else content[:200]
            
            summary_parts.append(f"{role}: {first_sentence}...")
        
        summary_content = "\n".join(summary_parts)
        
        return ConversationSummary(
            summary_type=focus or SummaryType.CONVERSATION_HISTORY,
            content=summary_content,
            turns_covered=[t.get("turn_number", i) for i, t in enumerate(conversation_turns, 1)],
            key_points=[],
            important_context={},
            token_count=len(summary_content.split())
        )
    
    async def summarize_with_map_reduce(
        self,
        conversation_turns: List[Dict[str, Any]],
        chunk_size: int = 10
    ) -> ConversationSummary:
        """
        Summarize long conversations using map-reduce strategy
        
        1. Split conversation into chunks
        2. Summarize each chunk (map)
        3. Summarize the summaries (reduce)
        
        This allows handling very long conversations (100+ turns)
        """
        if len(conversation_turns) <= chunk_size:
            return await self.summarize_conversation(conversation_turns)
        
        logger.info(f"Using map-reduce summarization for {len(conversation_turns)} turns")
        
        # Map: Summarize chunks
        chunk_summaries = []
        for i in range(0, len(conversation_turns), chunk_size):
            chunk = conversation_turns[i:i + chunk_size]
            chunk_summary = await self.summarize_conversation(chunk)
            chunk_summaries.append(chunk_summary)
        
        # Reduce: Summarize the summaries
        summary_texts = [
            {"role": "assistant", "content": cs.content, "turn_number": i}
            for i, cs in enumerate(chunk_summaries)
        ]
        
        final_summary = await self.summarize_conversation(summary_texts)
        
        # Combine key points and context from all chunk summaries
        all_key_points = []
        all_context = {
            "files_mentioned": set(),
            "functions_mentioned": set(),
            "errors_encountered": [],
            "decisions_made": []
        }
        
        for cs in chunk_summaries:
            all_key_points.extend(cs.key_points)
            all_context["files_mentioned"].update(cs.important_context.get("files_mentioned", []))
            all_context["functions_mentioned"].update(cs.important_context.get("functions_mentioned", []))
            all_context["errors_encountered"].extend(cs.important_context.get("errors_encountered", []))
        
        # Convert sets back to lists
        all_context["files_mentioned"] = list(all_context["files_mentioned"])
        all_context["functions_mentioned"] = list(all_context["functions_mentioned"])
        
        final_summary.key_points = all_key_points[:15]  # Top 15 key points
        final_summary.important_context = all_context
        final_summary.turns_covered = list(range(1, len(conversation_turns) + 1))
        
        logger.info(f"Map-reduce summarization complete: {len(conversation_turns)} turns -> {final_summary.token_count} tokens")
        
        return final_summary


class HierarchicalMemoryManager:
    """
    Hierarchical memory management system
    
    Implements a three-tier memory system similar to production coding assistants:
    - Working Memory: Current active context (full detail)
    - Short-term Memory: Recent conversation (summaries)
    - Long-term Memory: Historical context (compressed)
    """
    
    def __init__(
        self,
        working_memory_turns: int = 5,
        short_term_turns: int = 20,
        summarizer: Optional[LLMSummarizer] = None
    ):
        """
        Initialize hierarchical memory
        
        Args:
            working_memory_turns: Number of recent turns to keep in full detail
            short_term_turns: Number of turns to keep in short-term memory
            summarizer: LLM summarizer for generating compressed memories
        """
        self.working_memory_turns = working_memory_turns
        self.short_term_turns = short_term_turns
        self.summarizer = summarizer
        
        self.working_memory: List[Dict[str, Any]] = []
        self.short_term_memory: List[ConversationSummary] = []
        self.long_term_memory: Optional[ConversationSummary] = None
    
    async def add_turn(self, turn: Dict[str, Any]) -> None:
        """
        Add a new conversation turn and manage memory tiers
        
        Args:
            turn: Conversation turn to add
        """
        # Add to working memory
        self.working_memory.append(turn)
        
        # If working memory exceeds limit, compress to short-term
        if len(self.working_memory) > self.working_memory_turns:
            await self._compress_to_short_term()
    
    async def _compress_to_short_term(self) -> None:
        """Move oldest working memory to short-term memory"""
        # Take oldest turns from working memory
        turns_to_compress = self.working_memory[:-self.working_memory_turns]
        self.working_memory = self.working_memory[-self.working_memory_turns:]
        
        if not turns_to_compress:
            return
        
        # Summarize if summarizer available
        if self.summarizer:
            summary = await self.summarizer.summarize_conversation(turns_to_compress)
            self.short_term_memory.append(summary)
        else:
            # Fallback: store turns directly
            summary = ConversationSummary(
                summary_type=SummaryType.CONVERSATION_HISTORY,
                content="\n".join([t.get("content", "")[:200] for t in turns_to_compress]),
                turns_covered=[t.get("turn_number", i) for i, t in enumerate(turns_to_compress)],
                key_points=[],
                important_context={},
                token_count=0
            )
            self.short_term_memory.append(summary)
        
        # If short-term memory exceeds limit, compress to long-term
        total_short_term_turns = sum(len(s.turns_covered) for s in self.short_term_memory)
        if total_short_term_turns > self.short_term_turns:
            await self._compress_to_long_term()
    
    async def _compress_to_long_term(self) -> None:
        """Compress short-term memory to long-term memory"""
        if not self.short_term_memory:
            return
        
        # Take half of short-term memory for compression
        mid_point = len(self.short_term_memory) // 2
        summaries_to_compress = self.short_term_memory[:mid_point]
        self.short_term_memory = self.short_term_memory[mid_point:]
        
        if not summaries_to_compress:
            return
        
        # Combine summaries
        combined_content = "\n\n---\n\n".join([s.content for s in summaries_to_compress])
        
        # If we have a summarizer, create a summary of summaries
        if self.summarizer:
            summary_turns = [
                {"role": "assistant", "content": s.content, "turn_number": i}
                for i, s in enumerate(summaries_to_compress)
            ]
            long_term_summary = await self.summarizer.summarize_conversation(summary_turns)
        else:
            long_term_summary = ConversationSummary(
                summary_type=SummaryType.CONVERSATION_HISTORY,
                content=combined_content[:5000],  # Limit size
                turns_covered=[],
                key_points=[],
                important_context={},
                token_count=len(combined_content.split())
            )
        
        # Merge with existing long-term memory
        if self.long_term_memory:
            self.long_term_memory.content += "\n\n" + long_term_summary.content
            self.long_term_memory.turns_covered.extend(long_term_summary.turns_covered)
        else:
            self.long_term_memory = long_term_summary
    
    def get_full_context(self) -> Dict[str, Any]:
        """
        Get full context across all memory tiers
        
        Returns:
            Dictionary with working, short-term, and long-term memory
        """
        return {
            "working_memory": self.working_memory,
            "short_term_memory": [s.content for s in self.short_term_memory],
            "long_term_memory": self.long_term_memory.content if self.long_term_memory else None,
            "memory_stats": {
                "working_turns": len(self.working_memory),
                "short_term_summaries": len(self.short_term_memory),
                "has_long_term": self.long_term_memory is not None
            }
        }
    
    def get_context_for_llm(self) -> str:
        """
        Get formatted context string for LLM
        
        Returns:
            Formatted string with all relevant context
        """
        parts = []
        
        # Add long-term memory if exists
        if self.long_term_memory:
            parts.append(f"=== HISTORICAL CONTEXT ===\n{self.long_term_memory.content}\n")
        
        # Add short-term memory
        if self.short_term_memory:
            parts.append("=== RECENT CONTEXT ===")
            for i, summary in enumerate(self.short_term_memory, 1):
                parts.append(f"\nSummary {i}:\n{summary.content}")
            parts.append("")
        
        # Add working memory (full detail)
        if self.working_memory:
            parts.append("=== CURRENT CONVERSATION ===")
            for turn in self.working_memory:
                role = turn.get("role", "unknown")
                content = turn.get("content", "")
                parts.append(f"\n{role.upper()}: {content}")
        
        return "\n".join(parts)

