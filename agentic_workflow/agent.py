"""
Agent framework for orchestrating tool calls using LLM
"""

import json
import logging
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from .tool_registry import get_registry
from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """State management for the agent"""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    current_step: str = ""
    completed_steps: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class Agent:
    """Agent that uses LLM to orchestrate tool calls"""
    
    def __init__(self, config: Config):
        self.config = config
        self.registry = get_registry()
        self.state = AgentState()
        self.session = requests.Session()
    
    def _call_llm(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call the LLM via OpenAI-compatible API"""
        url = f"{self.config.vllm_url}/v1/chat/completions"
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        try:
            response = self.session.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def _process_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Process a tool call from the LLM"""
        function_name = tool_call["function"]["name"]
        
        try:
            arguments = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError:
            arguments = {}
        
        logger.info(f"Processing tool call: {function_name} with args: {arguments}")
        
        result = self.registry.execute(function_name, arguments)
        self.state.tool_results.append({
            "tool": function_name,
            "arguments": arguments,
            "result": result
        })
        
        return result
    
    def execute_workflow(self, user_input: str, max_iterations: int = 20) -> Dict[str, Any]:
        """
        Execute a workflow based on user input
        
        Args:
            user_input: User's request/instructions
            max_iterations: Maximum number of LLM iterations
            
        Returns:
            Dictionary with workflow results
        """
        # Initialize conversation
        system_prompt = """You are an intelligent agent that orchestrates research paper collection workflows.
You have access to the following tools:
1. crossref_search - Search CrossRef API for papers
2. combine_and_deduplicate - Combine search results and remove duplicates
3. download_articles - Download article XMLs from DOIs
4. build_database - Extract metadata from downloaded XMLs
5. organize_xmls - Organize XMLs into combined directory

Your task is to execute these tools in the correct sequence based on user requirements.
Always provide clear reasoning for your tool choices and handle errors gracefully."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        tools = self.registry.get_all_schemas()
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Agent iteration {iteration}/{max_iterations}")
            
            # Call LLM
            try:
                response = self._call_llm(messages, tools)
            except Exception as e:
                error_msg = f"LLM call failed: {str(e)}"
                logger.error(error_msg)
                self.state.errors.append(error_msg)
                break
            
            # Extract response
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])
            
            # Add assistant message to history
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls if tool_calls else None
            })
            
            # If no tool calls, we're done
            if not tool_calls:
                messages.append({
                    "role": "user",
                    "content": "Workflow completed. Provide a summary of what was accomplished."
                })
                final_response = self._call_llm(messages, tools)
                final_content = final_response.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                return {
                    "success": True,
                    "summary": final_content,
                    "tool_results": self.state.tool_results,
                    "errors": self.state.errors
                }
            
            # Process tool calls
            tool_messages = []
            for tool_call in tool_calls:
                result = self._process_tool_call(tool_call)
                
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "name": tool_call["function"]["name"],
                    "content": json.dumps({
                        "success": result["success"],
                        "result": result["result"],
                        "error": result["error"]
                    })
                })
            
            messages.extend(tool_messages)
        
        # Max iterations reached
        return {
            "success": False,
            "summary": "Maximum iterations reached",
            "tool_results": self.state.tool_results,
            "errors": self.state.errors + ["Maximum iterations reached"]
        }
    
    def reset_state(self):
        """Reset agent state"""
        self.state = AgentState()

