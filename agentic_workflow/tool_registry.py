"""
Tool registry for managing and executing tools
"""

import json
import logging
from typing import Dict, List, Callable, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing tools and their schemas"""
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, func: Callable, description: str, 
                 parameters: Dict[str, Any]) -> None:
        """
        Register a tool with the registry
        
        Args:
            name: Tool name
            func: Tool function
            description: Tool description
            parameters: JSON schema for parameters
        """
        self.tools[name] = {
            "function": func,
            "description": description,
            "parameters": parameters
        }
        logger.info(f"Registered tool: {name}")
    
    def get_function_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Get function calling schema for a tool"""
        if name not in self.tools:
            return None
        
        tool = self.tools[name]
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": tool["description"],
                "parameters": {
                    "type": "object",
                    "properties": tool["parameters"].get("properties", {}),
                    "required": tool["parameters"].get("required", [])
                }
            }
        }
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get function calling schemas for all registered tools"""
        return [self.get_function_schema(name) for name in self.tools.keys()]
    
    def _convert_arguments(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert and validate arguments based on tool schema
        
        Args:
            name: Tool name
            arguments: Raw arguments from LLM
            
        Returns:
            Converted arguments with proper types
        """
        if name not in self.tools:
            return arguments
        
        tool = self.tools[name]
        schema = tool["parameters"]
        properties = schema.get("properties", {})
        converted = {}
        
        for key, value in arguments.items():
            if key not in properties:
                # Keep unknown arguments as-is
                converted[key] = value
                continue
            
            prop_schema = properties[key]
            prop_type = prop_schema.get("type")
            
            # Handle string representations of types
            if isinstance(value, str):
                # Try to parse JSON strings
                if value.startswith('[') or value.startswith('{'):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        # Try using ast.literal_eval for Python literal strings
                        try:
                            import ast
                            value = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            pass
            
            # Type conversion based on schema
            if prop_type == "integer":
                try:
                    converted[key] = int(value)
                except (ValueError, TypeError):
                    converted[key] = value
            elif prop_type == "number":
                try:
                    converted[key] = float(value)
                except (ValueError, TypeError):
                    converted[key] = value
            elif prop_type == "array":
                # Ensure it's a list
                if not isinstance(value, list):
                    if isinstance(value, str):
                        # Try to parse as JSON
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            # Split by comma if it's a simple string
                            value = [v.strip() for v in value.split(',')]
                    else:
                        value = [value]
                
                # Special handling for date_ranges - convert list of lists to list of tuples
                if key == "date_ranges":
                    converted_ranges = []
                    for item in value:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            # Convert to tuple (even if already a tuple, ensure it's a tuple)
                            converted_ranges.append(tuple(item))
                        elif isinstance(item, str):
                            # Try to parse as JSON
                            try:
                                parsed = json.loads(item)
                                if isinstance(parsed, (list, tuple)) and len(parsed) == 2:
                                    converted_ranges.append(tuple(parsed))
                            except json.JSONDecodeError:
                                logger.warning(f"Could not parse date_range item: {item}")
                    # Always use converted_ranges if we processed any items
                    converted[key] = converted_ranges
                else:
                    converted[key] = value
            else:
                converted[key] = value
        
        return converted
    
    def execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with given arguments
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Dictionary with 'success', 'result', and 'error' keys
        """
        if name not in self.tools:
            error_msg = f"Tool '{name}' not found in registry"
            logger.error(error_msg)
            return {
                "success": False,
                "result": None,
                "error": error_msg
            }
        
        tool = self.tools[name]
        func = tool["function"]
        
        try:
            # Convert and validate arguments
            converted_args = self._convert_arguments(name, arguments)
            logger.info(f">>> Executing tool: {name} <<<")
            logger.debug(f"Tool {name} arguments: {converted_args}")
            result = func(**converted_args)
            return {
                "success": True,
                "result": result,
                "error": None
            }
        except Exception as e:
            error_msg = f"Error executing tool '{name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "result": None,
                "error": error_msg
            }
    
    def list_tools(self) -> List[str]:
        """List all registered tool names"""
        return list(self.tools.keys())


# Global registry instance
_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance"""
    return _registry


def tool(name: str, description: str, parameters: Dict[str, Any]):
    """
    Decorator to register a function as a tool
    
    Args:
        name: Tool name
        description: Tool description
        parameters: JSON schema for parameters
    """
    def decorator(func: Callable) -> Callable:
        _registry.register(name, func, description, parameters)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

