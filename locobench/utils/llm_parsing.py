"""
Advanced LLM Response Parsing for LoCoBench

This module provides robust parsing capabilities for LLM responses,
handling JSON extraction, code block parsing, and intelligent fallbacks.
"""

import json
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMResponseParser:
    """Advanced parser for LLM responses with multiple fallback strategies"""
    
    def __init__(self):
        # Patterns for different response formats
        self.json_patterns = [
            # Pattern 1: ```json blocks - fixed to capture everything between delimiters
            r'```json\s*\n?(.*?)\n?\s*```',
            
            # Pattern 2: Generic ``` blocks - fixed to capture everything between delimiters
            r'```\s*\n?(.*?)\n?\s*```',
            
            # Pattern 3: Specific files pattern - non-greedy
            r'(\{[^{}]*?"files"[^{}]*?\{.*?\}[^{}]*?\})',
            
            # Pattern 4: Any JSON-like structure - more conservative
            r'(\{(?:[^{}]|\{[^{}]*\})*\})',
            
            # Pattern 5: Multiline JSON without markdown
            r'(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})'
        ]
        
        self.code_block_patterns = {
            'go': r'```go\s*(.*?)\s*```',
            'python': r'```python\s*(.*?)\s*```', 
            'javascript': r'```(?:js|javascript)\s*(.*?)\s*```',
            'typescript': r'```(?:ts|typescript)\s*(.*?)\s*```',
            'rust': r'```rust\s*(.*?)\s*```',
            'java': r'```java\s*(.*?)\s*```',
            'cpp': r'```(?:cpp|c\+\+)\s*(.*?)\s*```',
            'generic': r'```\w*\s*(.*?)\s*```'
        }
        
        # Language-specific indicators
        self.language_indicators = {
            'go': ['package main', 'func main', 'import "', 'type ', 'func '],
            'python': ['def ', 'import ', 'from ', 'class ', '__init__'],
            'javascript': ['function ', 'const ', 'let ', 'var ', '=>'],
            'typescript': ['interface ', 'type ', 'function ', 'const ', '=>'],
            'rust': ['fn ', 'use ', 'mod ', 'struct ', 'impl '],
            'java': ['public class', 'private ', 'public ', 'static '],
            'cpp': ['#include', 'int main', 'class ', 'namespace ']
        }

    def _is_viable_result(self, files: Dict[str, str]) -> bool:
        """Check if the extracted files are viable (not suspiciously short)"""
        if not files:
            return False
        
        # Check if any file has reasonable content length
        min_viable_length = 50  # Minimum characters for viable code
        
        for filename, content in files.items():
            # For code files, expect at least some meaningful content
            if len(content) >= min_viable_length:
                return True
        
        # If all files are very short, this might be a parsing artifact
        return False

    def parse(self, response: str, expected_language: str = 'python') -> Dict[str, str]:
        """Parse LLM response with multiple fallback strategies"""
        
        # Strategy 1: Structured JSON extraction
        structured_result = self._extract_structured_json(response)
        if structured_result and self._is_viable_result(structured_result):
            logger.info(f"✅ Successfully extracted {len(structured_result)} files from structured JSON")
            return structured_result
            
        # Strategy 2: JSON-like structures with viability check
        json_like_result = self._extract_code_from_json_like(response)
        if json_like_result and self._is_viable_result(json_like_result):
            logger.info(f"✅ Successfully extracted {len(json_like_result)} files from JSON-like structure")
            return json_like_result
            
        # Strategy 3: Markdown code blocks
        code_blocks_result = self._extract_code_blocks(response, expected_language)
        if code_blocks_result and self._is_viable_result(code_blocks_result):
            logger.info(f"✅ Successfully extracted {len(code_blocks_result)} code blocks")
            return code_blocks_result
            
        # Strategy 4: Intelligent text parsing
        text_parsing_result = self._intelligent_text_parsing(response, expected_language)
        if text_parsing_result and self._is_viable_result(text_parsing_result):
            logger.info("✅ Successfully parsed code from text analysis")
            return text_parsing_result
        
        # Strategy 5: Manual extraction for extremely long JSON
        manual_result = self._extract_files_manually(response)
        if manual_result and self._is_viable_result(manual_result):
            logger.info("✅ Successfully extracted files manually")
            return manual_result
            
        # No fallbacks - if parsing fails, let it fail clearly
        logger.error("❌ All parsing strategies failed - no fallbacks allowed")
        return None

    def _extract_structured_json(self, response: str) -> Optional[Dict[str, str]]:
        """Extract properly structured JSON responses with enhanced o3 support"""
        
        # Enhanced patterns specifically for o3's verbose responses
        enhanced_patterns = [
            # Pattern for o3's verbose responses with explanations before JSON
            r'(?:```json\s*\n?|Here\'s the|I\'ll provide|The solution|```\s*\n?)(\{.*?\})\s*(?:```|$)',
            
            # Multiple JSON blocks - take the largest one
            r'```json\s*\n?(.*?)\n?\s*```',
            
            # JSON embedded in explanatory text
            r'solution.*?(\{[^{}]*?"files"[^{}]*?\{.*?\}[^{}]*?\})',
            
            # Direct JSON without markdown
            r'(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})',
            
            # JSON at the end of response after explanations
            r'(?:final|solution|result).*?(\{.*?\})\s*$'
        ]
        
        # Combine enhanced patterns with existing ones
        all_patterns = enhanced_patterns + self.json_patterns
        
        best_result = None
        best_score = 0
        
        for pattern in all_patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                try:
                    # Enhanced JSON extraction with multiple strategies
                    extracted_json = None
                    
                    # Strategy 1: Direct extraction for complete objects
                    if pattern.startswith(r'```json') or 'files' in pattern:
                        extracted_json = self._extract_json_object(match)
                        if not extracted_json:
                            extracted_json = self._extract_json_from_verbose_response(match)
                    else:
                        extracted_json = self._clean_json_string(match)
                    
                    if not extracted_json:
                        continue
                        
                    # Try progressive JSON cleaning strategies
                    for cleaning_strategy in [self._clean_json_string, self._aggressive_json_cleaning]:
                        try:
                            cleaned_json = cleaning_strategy(extracted_json)
                            data = json.loads(cleaned_json)
                            
                            # Look for files in various possible keys
                            files = self._extract_files_from_data(data)
                            if files:
                                # Score this result based on content quality
                                score = self._score_json_result(files, data)
                                if score > best_score:
                                    best_result = files
                                    best_score = score
                                    
                        except json.JSONDecodeError:
                            continue
                        
                except Exception as e:
                    logger.debug(f"JSON extraction failed: {e}")
                    continue
        
        return best_result
    
    def _extract_json_object(self, text: str) -> Optional[str]:
        """Extract the first complete JSON object from text"""
        text = text.strip()
        if not text.startswith('{'):
            return None
        
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text[:i+1]
        
        return None

    def _extract_json_from_verbose_response(self, text: str) -> Optional[str]:
        """Extract JSON from o3's verbose responses with explanations"""
        text = text.strip()
        
        # Look for the largest JSON object in the text
        json_candidates = []
        
        # Find all potential JSON objects
        brace_positions = []
        for i, char in enumerate(text):
            if char == '{':
                brace_positions.append(i)
        
        for start_pos in brace_positions:
            extracted = self._extract_json_object(text[start_pos:])
            if extracted and len(extracted) > 50:  # Minimum meaningful size
                json_candidates.append(extracted)
        
        # Return the largest valid JSON object
        if json_candidates:
            return max(json_candidates, key=len)
        
        return None

    def _aggressive_json_cleaning(self, text: str) -> str:
        """Aggressive JSON cleaning for o3's complex responses"""
        text = text.strip()
        
        # Remove common o3 verbose patterns
        patterns_to_remove = [
            r'^Here\'s.*?:\s*',  # "Here's the solution:"
            r'^I\'ll.*?:\s*',    # "I'll provide:"
            r'^The solution.*?:\s*',  # "The solution is:"
            r'\n\s*//.*$',       # End-of-line comments
            r'\n\s*\*.*$',       # Markdown bullet points
            r'```\s*$',          # Trailing markdown
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        # Fix common JSON issues in o3 responses
        fixes = [
            (r',\s*}', '}'),  # Remove trailing commas
            (r',\s*]', ']'),  # Remove trailing commas in arrays
            (r'"\s*\n\s*"', '" "'),  # Fix line breaks in strings
            (r'\\n', '\\\\n'),  # Escape newlines properly
            (r'([^\\])"([^"]*?)"([^:])', r'\1"\2"\3'),  # Fix unescaped quotes
        ]
        
        for pattern, replacement in fixes:
            text = re.sub(pattern, replacement, text)
        
        return text

    def _score_json_result(self, files: Dict[str, str], data: Dict[str, Any]) -> int:
        """Score a JSON parsing result to pick the best one"""
        score = 0
        
        # More files is generally better
        score += len(files) * 10
        
        # More content is generally better
        total_content = sum(len(code) for code in files.values())
        score += min(total_content // 100, 100)  # Cap content score
        
        # Prefer solutions with proper structure
        if 'approach' in data:
            score += 20
        if 'files' in data:
            score += 30
        if 'explanation' in data:
            score += 10
            
        # Penalty for suspiciously short content
        if total_content < 50:
            score -= 50
            
        return score

    def _emergency_code_extraction(self, response: str, expected_language: str) -> Optional[Dict[str, str]]:
        """Emergency code extraction when all else fails"""
        
        # Try to find ANY code blocks in the response
        code_blocks = []
        
        # Look for markdown code blocks with any language
        markdown_pattern = r'```(\w*)\s*(.*?)\s*```'
        matches = re.findall(markdown_pattern, response, re.DOTALL)
        
        for lang, code in matches:
            if code.strip() and len(code.strip()) > 20:
                filename = f"extracted_{len(code_blocks)}.{self._get_language_extension(lang or expected_language)}"
                code_blocks.append((filename, code.strip()))
        
        # If no markdown blocks, look for any substantial text that looks like code
        if not code_blocks:
            lines = response.split('\n')
            code_lines = []
            for line in lines:
                # Look for lines that look like code (have common code patterns)
                if any(pattern in line for pattern in ['{', '}', '()', 'function', 'class', 'def ', 'import ', '#include']):
                    code_lines.append(line)
            
            if code_lines and len('\n'.join(code_lines)) > 50:
                filename = f"extracted.{self._get_language_extension(expected_language)}"
                code_blocks.append((filename, '\n'.join(code_lines)))
        
        if code_blocks:
            return dict(code_blocks)
        
        return None

    def _get_language_extension(self, language: str) -> str:
        """Get file extension for a programming language"""
        extensions = {
            'python': 'py', 'go': 'go', 'javascript': 'js', 'typescript': 'ts',
            'rust': 'rs', 'java': 'java', 'cpp': 'cpp', 'c++': 'cpp',
            'c': 'c', 'php': 'php', 'ruby': 'rb', 'swift': 'swift'
        }
        return extensions.get(language.lower(), 'txt')

    def _create_placeholder_content(self, language: str, response: str) -> str:
        """Create meaningful placeholder content when parsing completely fails"""
        
        # Try to extract any meaningful text from the response
        response_summary = response[:200] if response else "No response received"
        
        templates = {
            'python': f'''# LoCoBench Parsing Fallback
# Original response parsing failed, this is a placeholder
"""
Response summary: {response_summary}...
"""

def placeholder_function():
    # Placeholder implementation
    print("LoCoBench placeholder - parsing failed")
    return "placeholder"

if __name__ == "__main__":
    placeholder_function()
''',
            'go': f'''// LoCoBench Parsing Fallback
// Original response parsing failed, this is a placeholder
package main

import "fmt"

/* Response summary: {response_summary}... */

func main() {{
    fmt.Println("LoCoBench placeholder - parsing failed")
}}
''',
            'javascript': f'''// LoCoBench Parsing Fallback
// Original response parsing failed, this is a placeholder
/* Response summary: {response_summary}... */

function placeholderFunction() {{
    console.log("LoCoBench placeholder - parsing failed");
    return "placeholder";
}}

if (typeof module !== 'undefined') {{
    module.exports = {{ placeholderFunction }};
}}
''',
            'rust': f'''// LoCoBench Parsing Fallback
// Original response parsing failed, this is a placeholder
/* Response summary: {response_summary}... */

fn main() {{
    println!("LoCoBench placeholder - parsing failed");
}}
''',
            'java': f'''// LoCoBench Parsing Fallback
// Original response parsing failed, this is a placeholder
/* Response summary: {response_summary}... */

public class PlaceholderMain {{
    public static void main(String[] args) {{
        System.out.println("LoCoBench placeholder - parsing failed");
    }}
}}
'''
        }
        
        return templates.get(language.lower(), f'// Placeholder for {language}\n// {response_summary}...')

    def _extract_code_from_json_like(self, response: str) -> Optional[Dict[str, str]]:
        """Extract code from JSON-like structures that might have parsing issues"""
        
        # Look for patterns like "filename.go": "code content"
        file_pattern = r'"([^"]+\.(?:go|py|js|ts|rs|java|cpp|h))"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
        matches = re.findall(file_pattern, response, re.DOTALL)
        
        if matches:
            files = {}
            for filename, code in matches:
                # Unescape the code content
                unescaped_code = self._unescape_code(code)
                files[filename] = unescaped_code
            return files
            
        # Look for key-value patterns without quotes
        kv_pattern = r'(\w+\.(?:go|py|js|ts|rs|java|cpp|h))\s*[:=]\s*`([^`]+)`'
        kv_matches = re.findall(kv_pattern, response, re.DOTALL)
        
        if kv_matches:
            return {filename: code for filename, code in kv_matches}
            
        return None

    def _extract_code_blocks(self, response: str, expected_language: str) -> Optional[Dict[str, str]]:
        """Extract code from markdown code blocks"""
        
        files = {}
        
        # Try language-specific pattern first
        if expected_language in self.code_block_patterns:
            pattern = self.code_block_patterns[expected_language]
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            
            for i, code in enumerate(matches):
                filename = f"solution_{i+1}.{self._get_file_extension(expected_language)}"
                files[filename] = code.strip()
        
        # Try generic code blocks
        if not files:
            pattern = self.code_block_patterns['generic']
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            
            for i, code in enumerate(matches):
                # Try to detect language from content
                detected_lang = self._detect_language(code)
                extension = self._get_file_extension(detected_lang or expected_language)
                filename = f"solution_{i+1}.{extension}"
                files[filename] = code.strip()
        
        return files if files else None

    def _intelligent_text_parsing(self, response: str, expected_language: str) -> Optional[Dict[str, str]]:
        """Parse code using intelligent text analysis"""
        
        # Look for file headers or separators
        file_separators = [
            r'(?:^|\n)(?:File|Filename):\s*([^\n]+)',
            r'(?:^|\n)#+\s*([^\n]+\.(?:go|py|js|ts|rs|java|cpp|h))',
            r'(?:^|\n)//\s*([^\n]+\.(?:go|py|js|ts|rs|java|cpp|h))',
            r'(?:^|\n)#\s*([^\n]+\.(?:go|py|js|ts|rs|java|cpp|h))'
        ]
        
        files = {}
        
        for separator_pattern in file_separators:
            matches = list(re.finditer(separator_pattern, response, re.IGNORECASE | re.MULTILINE))
            
            if matches:
                for i, match in enumerate(matches):
                    filename = match.group(1).strip()
                    start = match.end()
                    
                    # Find the end of this file's content
                    if i + 1 < len(matches):
                        end = matches[i + 1].start()
                    else:
                        end = len(response)
                    
                    code_content = response[start:end].strip()
                    
                    # Clean up the code content
                    code_content = self._clean_code_content(code_content)
                    
                    if code_content and len(code_content) > 10:  # Minimum viable code
                        files[filename] = code_content
        
        return files if files else None



    def _clean_json_string(self, json_str: str) -> str:
        """Clean and fix common JSON formatting issues"""
        
        # Remove leading/trailing whitespace
        json_str = json_str.strip()
        
        # Fix trailing commas before closing brackets/braces
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Enhanced cleaning for very long strings
        
        # Fix any double-escaped sequences that might confuse JSON parser
        json_str = json_str.replace('\\\\n', '\\n')  # Fix double-escaped newlines
        json_str = json_str.replace('\\\\t', '\\t')  # Fix double-escaped tabs
        json_str = json_str.replace('\\\\"', '\\"')  # Fix double-escaped quotes
        json_str = json_str.replace('\\\\r', '\\r')  # Fix double-escaped carriage returns
        
        # NEW: Handle problematic characters in very long strings
        # Fix unescaped quotes that aren't part of JSON structure
        # This is a more aggressive fix for malformed long strings
        try:
            # First, try to parse as-is
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            # If parsing fails, try to fix common issues
            
            # Try to identify if it's an unterminated string issue
            if "Unterminated string" in str(e):
                # More aggressive cleaning for very long content
                # Split by lines and try to fix string escaping issues
                lines = json_str.split('\n')
                fixed_lines = []
                
                for line in lines:
                    # Ensure all quotes in JSON string values are properly escaped
                    # This is a simplified fix - more sophisticated logic could be added
                    fixed_line = line
                    
                    # If we're inside a JSON string value (after :"), fix unescaped quotes
                    if '": "' in line and not line.strip().endswith('",') and not line.strip().endswith('"'):
                        # This might be a broken string - try to close it
                        if not fixed_line.rstrip().endswith('"'):
                            fixed_line = fixed_line.rstrip() + '"'
                    
                    fixed_lines.append(fixed_line)
                
                json_str = '\n'.join(fixed_lines)
        
        return json_str

    def _extract_files_from_data(self, data: Any) -> Optional[Dict[str, str]]:
        """Extract files from parsed JSON data with multi-session support"""
        
        if isinstance(data, dict):
            # Direct files key
            if 'files' in data:
                files_data = data['files']
                
                # Handle list of file objects: [{"filename": "...", "content": "..."}, ...]
                if isinstance(files_data, list):
                    return self._convert_file_list_to_dict(files_data)
                
                # Handle dictionary of files
                elif isinstance(files_data, dict):
                    # Check if this is a multi-session structure (session_1, session_2, etc.)
                    if self._is_multi_session_structure(files_data):
                        return self._flatten_multi_session_files(files_data)
                    
                    return files_data
            
            # Solution key containing files
            if 'solution' in data and isinstance(data['solution'], dict):
                if 'files' in data['solution']:
                    solution_files = data['solution']['files']
                    if self._is_multi_session_structure(solution_files):
                        return self._flatten_multi_session_files(solution_files)
                    return solution_files
                
                # Check if solution itself has multi-session structure
                if self._is_multi_session_structure(data['solution']):
                    return self._flatten_multi_session_files(data['solution'])
                    
                return data['solution']
            
            # Code key
            if 'code' in data and isinstance(data['code'], dict):
                code_files = data['code']
                if self._is_multi_session_structure(code_files):
                    return self._flatten_multi_session_files(code_files)
                return code_files
            
            # Check for session-based keys at the top level (session_1_files, session_2_files, etc.)
            session_files = {}
            regular_files = {}
            
            for key, value in data.items():
                if isinstance(value, dict):
                    key_lower = str(key).lower()
                    
                    # Check for session-based keys (more flexible detection)
                    if any(pattern in key_lower for pattern in ['session_', 'step_', 'phase_', 'stage_']):
                        # This looks like session-based structure
                        session_files[key] = value
                    # Check if it looks like a filename -> code mapping
                    elif any(k.endswith(('.go', '.py', '.js', '.ts', '.rs', '.java', '.cpp', '.h', '.txt', '.md', '.json', '.xml', '.yml', '.yaml', '.conf', '.cfg', '.ini', '.properties')) 
                           for k in value.keys() if isinstance(k, str)):
                        regular_files.update(value)
            
            # If we found session-based files, flatten them
            if session_files:
                flattened = self._flatten_multi_session_files(session_files)
                # Also include any regular files found
                flattened.update(regular_files)
                return flattened
            
            # If we found regular files, return them
            if regular_files:
                return regular_files
        
        return None
    
    def _is_multi_session_structure(self, files_dict: Dict[str, Any]) -> bool:
        """Check if the files dictionary has a multi-session structure"""
        if not isinstance(files_dict, dict):
            return False
            
        # Look for session-based keys (session_1, session_2, etc. or similar patterns)
        session_patterns = ['session_', 'step_', 'phase_', 'stage_']
        session_keys = []
        
        for key in files_dict.keys():
            key_lower = str(key).lower()
            if any(pattern in key_lower for pattern in session_patterns):
                session_keys.append(key)
            elif key_lower.startswith(('s1', 's2', 's3', 'stage1', 'stage2', 'stage3')):
                session_keys.append(key)
        
        # If we have session keys and they contain nested dictionaries, it's multi-session
        if session_keys:
            return any(isinstance(files_dict[key], dict) for key in session_keys)
        
        return False
    
    def _flatten_multi_session_files(self, files_dict: Dict[str, Any]) -> Dict[str, str]:
        """Flatten multi-session file structure into a single files dictionary"""
        flattened = {}
        
        for session_key, session_data in files_dict.items():
            if isinstance(session_data, dict):
                # Add files from this session, prefixing with session if there are conflicts
                for filename, content in session_data.items():
                    if isinstance(content, str):
                        # Use original filename if no conflict, otherwise prefix with session
                        final_filename = filename
                        if filename in flattened:
                            final_filename = f"{session_key}_{filename}"
                        flattened[final_filename] = content
                    elif isinstance(content, (dict, list)):
                        # Convert non-string content to JSON string
                        flattened[filename] = json.dumps(content, indent=2)
            elif isinstance(session_data, str):
                # Direct content (unusual but handle it)
                flattened[session_key] = session_data
        
        return flattened
    
    def _convert_file_list_to_dict(self, file_list: List[Any]) -> Dict[str, str]:
        """Convert list of file objects to filename -> content dictionary"""
        files_dict = {}
        
        for i, item in enumerate(file_list):
            if isinstance(item, dict):
                # Look for common filename/content key patterns
                filename = None
                content = None
                
                # Common patterns for filename
                for key in ['filename', 'file', 'name', 'path', 'file_path']:
                    if key in item:
                        filename = item[key]
                        break
                
                # Common patterns for content
                for key in ['content', 'code', 'source', 'text', 'body']:
                    if key in item:
                        content = item[key]
                        break
                
                # If we found both, add to dictionary
                if filename and content:
                    if isinstance(content, str):
                        files_dict[filename] = content
                    else:
                        # Convert non-string content to JSON
                        files_dict[filename] = json.dumps(content, indent=2)
                elif content:
                    # If no filename found, use index
                    files_dict[f"file_{i+1}.py"] = str(content)
            elif isinstance(item, str):
                # Direct string content, generate filename
                files_dict[f"file_{i+1}.py"] = item
        
        return files_dict

    def _unescape_code(self, code: str) -> str:
        """Unescape code content from JSON strings"""
        
        code = code.replace('\\"', '"')
        code = code.replace('\\n', '\n')
        code = code.replace('\\t', '\t')
        code = code.replace('\\r', '\r')
        code = code.replace('\\\\', '\\')
        
        return code

    def _detect_language(self, code: str) -> Optional[str]:
        """Detect programming language from code content"""
        
        code_lower = code.lower()
        
        for language, indicators in self.language_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in code_lower)
            if matches >= 2:  # Require at least 2 indicators
                return language
        
        return None

    def _get_file_extension(self, language: str) -> str:
        """Get file extension for a programming language"""
        
        extensions = {
            'go': 'go',
            'python': 'py',
            'javascript': 'js', 
            'typescript': 'ts',
            'rust': 'rs',
            'java': 'java',
            'cpp': 'cpp'
        }
        
        return extensions.get(language, 'go')  # Default to Go

    def _clean_code_content(self, content: str) -> str:
        """Clean extracted code content"""
        
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip empty lines at the beginning
            if not cleaned_lines and not line.strip():
                continue
            cleaned_lines.append(line)
        
        # Remove trailing empty lines
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        
        return '\n'.join(cleaned_lines)

    def _create_language_template(self, language: str, original_response: str) -> str:
        """Create a basic template for the given language"""
        
        templates = {
            'go': f"""package main

import "fmt"

// Generated solution based on LLM response
func main() {{
    fmt.Println("Solution implementation")
    // TODO: Implement functionality based on requirements
}}

/*
Original LLM Response:
{original_response[:1000]}...
*/
""",
            'python': f"""#!/usr/bin/env python3
\"\"\"
Generated solution based on LLM response
\"\"\"

def main():
    print("Solution implementation")
    # TODO: Implement functionality based on requirements

if __name__ == "__main__":
    main()

# Original LLM Response:
# {original_response[:500]}...
""",
            'javascript': f"""// Generated solution based on LLM response

function main() {{
    console.log("Solution implementation");
    // TODO: Implement functionality based on requirements
}}

main();

/*
Original LLM Response:
{original_response[:500]}...
*/
"""
        }
        
        return templates.get(language, templates['go'])

    def _extract_files_manually(self, response: str) -> Optional[Dict[str, str]]:
        """Manually extract files from JSON-like responses when regex fails
        
        This handles cases where the JSON contains extremely long strings
        that break regex patterns.
        """
        
        files = {}
        
        # Look for the files section
        files_start = response.find('"files"')
        if files_start == -1:
            return None
        
        # Find the opening brace of the files object
        brace_pos = response.find('{', files_start)
        if brace_pos == -1:
            return None
        
        # Parse character by character to extract files
        i = brace_pos + 1
        while i < len(response):
            # Skip whitespace
            while i < len(response) and response[i].isspace():
                i += 1
            
            if i >= len(response):
                break
                
            # Check for end of files object
            if response[i] == '}':
                break
            
            # Look for filename in quotes
            if response[i] == '"':
                # Extract filename
                i += 1  # Skip opening quote
                filename_start = i
                while i < len(response) and response[i] != '"':
                    if response[i] == '\\':
                        i += 2  # Skip escaped character
                    else:
                        i += 1
                
                if i >= len(response):
                    break
                    
                filename = response[filename_start:i]
                i += 1  # Skip closing quote
                
                # Skip whitespace and colon
                while i < len(response) and (response[i].isspace() or response[i] == ':'):
                    i += 1
                
                # Look for content in quotes
                if i < len(response) and response[i] == '"':
                    i += 1  # Skip opening quote
                    content_start = i
                    content_chars = []
                    
                    # Extract content, handling escape sequences
                    while i < len(response):
                        if response[i] == '\\' and i + 1 < len(response):
                            # Handle escape sequence
                            next_char = response[i + 1]
                            if next_char == 'n':
                                content_chars.append('\n')
                            elif next_char == 't':
                                content_chars.append('\t')
                            elif next_char == 'r':
                                content_chars.append('\r')
                            elif next_char == '"':
                                content_chars.append('"')
                            elif next_char == '\\':
                                content_chars.append('\\')
                            else:
                                content_chars.append(next_char)
                            i += 2
                        elif response[i] == '"':
                            # End of string
                            break
                        else:
                            content_chars.append(response[i])
                            i += 1
                    
                    if filename.endswith(('.go', '.py', '.js', '.ts', '.rs', '.java', '.cpp', '.h')):
                        content = ''.join(content_chars)
                        files[filename] = content
                    
                    i += 1  # Skip closing quote
            
            # Skip to next comma or end
            while i < len(response) and response[i] not in ',}':
                i += 1
            
            if i < len(response) and response[i] == ',':
                i += 1
        
        return files if files else None


# Convenience function for easy import
def parse_llm_response(response: str, expected_language: str = 'go') -> Optional[Dict[str, str]]:
    """Parse LLM response using the advanced parser - returns None if parsing fails"""
    parser = LLMResponseParser()
    return parser.parse(response, expected_language) 