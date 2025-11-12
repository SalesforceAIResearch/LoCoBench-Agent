"""
Simple Tools for Testing LoCoBench-Agent

This module provides basic tools for testing the agent system,
including echo and calculator tools.
"""

import logging
import math
from typing import Any, Dict

from ..core.tool_registry import Tool, ToolCategory, ToolParameter, tool_function

logger = logging.getLogger(__name__)


class EchoTool(Tool):
    """
    Simple echo tool for testing agent communication
    """
    
    def __init__(self, name: str = "echo"):
        super().__init__(
            name=name,
            description="Simple tool that echoes back messages with optional transformations",
            category=ToolCategory.COMMUNICATION
        )
    
    @tool_function(
        description="Echo back a message",
        parameters=[
            ToolParameter("message", "string", "Message to echo back", required=True),
            ToolParameter("uppercase", "boolean", "Convert to uppercase", required=False, default=False),
            ToolParameter("prefix", "string", "Prefix to add to the message", required=False, default="")
        ],
        returns="The echoed message with any requested transformations",
        category=ToolCategory.COMMUNICATION
    )
    def echo(self, message: str, uppercase: bool = False, prefix: str = "") -> str:
        """Echo back a message with optional transformations"""
        result = message
        
        if uppercase:
            result = result.upper()
        
        if prefix:
            result = f"{prefix}{result}"
        
        logger.debug(f"Echo: {message} -> {result}")
        return result
    
    @tool_function(
        description="Repeat a message multiple times",
        parameters=[
            ToolParameter("message", "string", "Message to repeat", required=True),
            ToolParameter("count", "integer", "Number of times to repeat (max 10)", required=True),
            ToolParameter("separator", "string", "Separator between repetitions", required=False, default=" ")
        ],
        returns="The message repeated the specified number of times",
        category=ToolCategory.COMMUNICATION
    )
    def repeat(self, message: str, count: int, separator: str = " ") -> str:
        """Repeat a message multiple times"""
        if count <= 0:
            return ""
        
        if count > 10:
            raise ValueError("Maximum repeat count is 10")
        
        result = separator.join([message] * count)
        logger.debug(f"Repeat: {message} x{count} -> {result}")
        return result


class CalculatorTool(Tool):
    """
    Simple calculator tool for basic mathematical operations
    """
    
    def __init__(self, name: str = "calculator"):
        super().__init__(
            name=name,
            description="Tool for performing basic mathematical calculations",
            category=ToolCategory.ANALYSIS
        )
    
    @tool_function(
        description="Add two numbers",
        parameters=[
            ToolParameter("a", "number", "First number", required=True),
            ToolParameter("b", "number", "Second number", required=True)
        ],
        returns="Sum of the two numbers",
        category=ToolCategory.ANALYSIS
    )
    def add(self, a: float, b: float) -> float:
        """Add two numbers"""
        result = a + b
        logger.debug(f"Add: {a} + {b} = {result}")
        return result
    
    @tool_function(
        description="Subtract two numbers",
        parameters=[
            ToolParameter("a", "number", "First number", required=True),
            ToolParameter("b", "number", "Second number", required=True)
        ],
        returns="Difference of the two numbers",
        category=ToolCategory.ANALYSIS
    )
    def subtract(self, a: float, b: float) -> float:
        """Subtract two numbers"""
        result = a - b
        logger.debug(f"Subtract: {a} - {b} = {result}")
        return result
    
    @tool_function(
        description="Multiply two numbers",
        parameters=[
            ToolParameter("a", "number", "First number", required=True),
            ToolParameter("b", "number", "Second number", required=True)
        ],
        returns="Product of the two numbers",
        category=ToolCategory.ANALYSIS
    )
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers"""
        result = a * b
        logger.debug(f"Multiply: {a} * {b} = {result}")
        return result
    
    @tool_function(
        description="Divide two numbers",
        parameters=[
            ToolParameter("a", "number", "Dividend", required=True),
            ToolParameter("b", "number", "Divisor", required=True)
        ],
        returns="Quotient of the division",
        category=ToolCategory.ANALYSIS
    )
    def divide(self, a: float, b: float) -> float:
        """Divide two numbers"""
        if b == 0:
            raise ValueError("Division by zero is not allowed")
        
        result = a / b
        logger.debug(f"Divide: {a} / {b} = {result}")
        return result
    
    @tool_function(
        description="Calculate the power of a number",
        parameters=[
            ToolParameter("base", "number", "Base number", required=True),
            ToolParameter("exponent", "number", "Exponent", required=True)
        ],
        returns="Result of base raised to the power of exponent",
        category=ToolCategory.ANALYSIS
    )
    def power(self, base: float, exponent: float) -> float:
        """Calculate the power of a number"""
        result = math.pow(base, exponent)
        logger.debug(f"Power: {base} ^ {exponent} = {result}")
        return result
    
    @tool_function(
        description="Calculate the square root of a number",
        parameters=[
            ToolParameter("number", "number", "Number to find square root of", required=True)
        ],
        returns="Square root of the number",
        category=ToolCategory.ANALYSIS
    )
    def sqrt(self, number: float) -> float:
        """Calculate the square root of a number"""
        if number < 0:
            raise ValueError("Cannot calculate square root of negative number")
        
        result = math.sqrt(number)
        logger.debug(f"Sqrt: √{number} = {result}")
        return result
    
    @tool_function(
        description="Evaluate a mathematical expression",
        parameters=[
            ToolParameter("expression", "string", "Mathematical expression to evaluate (basic operations only)", required=True)
        ],
        returns="Result of the mathematical expression",
        category=ToolCategory.ANALYSIS
    )
    def evaluate(self, expression: str) -> float:
        """Evaluate a mathematical expression (safely)"""
        # Whitelist of allowed characters and functions
        allowed_chars = set('0123456789+-*/()., ')
        allowed_functions = {'abs', 'min', 'max', 'round'}
        
        # Basic security check
        if not all(c in allowed_chars or c.isalpha() for c in expression):
            raise ValueError("Expression contains invalid characters")
        
        # Check for dangerous patterns
        dangerous_patterns = ['import', 'exec', 'eval', '__', 'open', 'file']
        expression_lower = expression.lower()
        if any(pattern in expression_lower for pattern in dangerous_patterns):
            raise ValueError("Expression contains potentially dangerous operations")
        
        try:
            # Create a safe namespace with only basic math functions
            safe_namespace = {
                '__builtins__': {},
                'abs': abs,
                'min': min,
                'max': max,
                'round': round,
                'pow': pow,
                'sqrt': math.sqrt,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'pi': math.pi,
                'e': math.e
            }
            
            result = eval(expression, safe_namespace, {})
            
            if not isinstance(result, (int, float)):
                raise ValueError("Expression must evaluate to a number")
            
            logger.debug(f"Evaluate: {expression} = {result}")
            return float(result)
            
        except Exception as e:
            logger.error(f"Error evaluating expression '{expression}': {e}")
            raise ValueError(f"Invalid mathematical expression: {e}")


# Convenience function to create default tools
def create_default_tools() -> Dict[str, Tool]:
    """Create a set of default tools for testing"""
    return {
        "echo": EchoTool(),
        "calculator": CalculatorTool(),
        "file_system": None  # Will be created with appropriate configuration
    }
