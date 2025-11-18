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
            logger.info(f"Executing tool: {name} with arguments: {arguments}")
            result = func(**arguments)
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

