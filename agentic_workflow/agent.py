"""
Agent framework for orchestrating tool calls using LLM
"""

import json
import logging
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from urllib.parse import urlparse

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
    reasoning_history: List[Dict[str, Any]] = field(default_factory=list)  # Store reasoning steps


class Agent:
    """Agent that uses LLM to orchestrate tool calls"""
    
    def __init__(self, config: Config):
        self.config = config
        self.registry = get_registry()
        self.state = AgentState()
        self.session = requests.Session()

    def _normalize_openai_base_url(self) -> str:
        base = (self.config.openai_base_url or "").rstrip("/")
        if not base:
            return "https://api.openai.com/v1"
        parsed = urlparse(base)
        path = parsed.path.rstrip("/")
        for suffix in ("/responses", "/chat/completions"):
            if path.endswith(suffix):
                path = path[: -len(suffix)]
                path = path.rstrip("/")
        if "/v1" in path:
            path = path[: path.index("/v1") + 3]
        else:
            path = f"{path}/v1" if path else "/v1"
        normalized = parsed._replace(path=path, params="", query="", fragment="").geturl().rstrip("/")
        return normalized

    def _tools_for_responses_api(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Chat Completions tool format (function nested) to Responses API format (top-level name, description, parameters)."""
        out = []
        for t in tools:
            if t.get("type") != "function" or "function" not in t:
                out.append(t)
                continue
            fn = t["function"]
            out.append({
                "type": "function",
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}, "required": []}),
            })
        return out

    def _extract_responses_text(self, response: Dict[str, Any]) -> str:
        output_text = response.get("output_text")
        if output_text:
            return output_text
        text_parts = []
        for item in response.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text_parts.append(content.get("text", ""))
        return "\n".join(text_parts).strip()
    
    def _compress_messages(self, messages: List[Dict[str, Any]], max_keep: int = 5) -> List[Dict[str, Any]]:
        """
        Compress conversation history using the LLM to fit within token limits
        
        Args:
            messages: Full conversation history
            max_keep: Number of most recent messages to keep as-is
            
        Returns:
            Compressed message history
        """
        # Ensure messages is a list
        if not messages:
            return []
        if len(messages) <= max_keep:
            return messages
        
        # Keep system message and most recent messages
        system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
        recent_messages = messages[-max_keep:] if len(messages) > max_keep else messages
        old_messages = messages[1:-max_keep] if system_msg and len(messages) > max_keep else (messages[:-max_keep] if len(messages) > max_keep else [])
        
        if not old_messages:
            return messages
        
        # Extract key information from old messages for summarization
        tool_calls_summary = []
        for msg in old_messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    tool_calls_summary.append(f"Called {tc.get('function', {}).get('name', 'unknown')}")
            elif msg.get("role") == "tool":
                try:
                    result = json.loads(msg.get("content", "{}"))
                    tool_name = msg.get("name", "unknown")
                    success = result.get("success", False)
                    tool_calls_summary.append(f"{tool_name}: {'success' if success else 'failed'}")
                except:
                    pass
        
        # Create a concise summary
        summary_parts = []
        if tool_calls_summary:
            summary_parts.append(f"Tools executed: {', '.join(tool_calls_summary[-5:])}")  # Last 5 tool calls
        
        # Ask LLM to summarize conversation (with reduced max_tokens to avoid recursion)
        compress_prompt = """Summarize this conversation in 2-3 sentences. Focus on:
- What workflow step we're on
- What tools have been called
- What needs to happen next

Be very brief."""
        
        # Prepare messages for compression (limit size to avoid token issues)
        messages_to_summarize = []
        for msg in old_messages:
            if msg.get("role") in ["user", "assistant"]:
                content = msg.get("content", "")
                # Truncate very long messages
                if len(content) > 500:
                    content = content[:500] + "..."
                messages_to_summarize.append(f"{msg.get('role')}: {content}")
        
        compress_messages = [
            {"role": "system", "content": compress_prompt},
            {"role": "user", "content": "\n".join(messages_to_summarize[-10:])}  # Last 10 messages only
        ]
        
        try:
            # Use a simple call without tools and with reduced max_tokens
            if self.config.llm_provider == "olmo":
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.config.openrouter_api_key}",
                    "HTTP-Referer": "https://github.com/matra-research",
                    "X-Title": "MatRA Research Agent"
                }
                model = self.config.openrouter_model
            elif self.config.llm_provider == "openai":
                base_url = self._normalize_openai_base_url()
                url = f"{base_url}/responses"
                headers = {
                    "Authorization": f"Bearer {self.config.openai_api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                model = self.config.openai_model
            else:
                url = f"{self.config.vllm_url}/v1/chat/completions"
                headers = {}
                model = self.config.model_name
            
            if self.config.llm_provider == "openai":
                # Responses API: gpt-5-nano and some models do not support temperature
                payload = {
                    "model": model,
                    "input": compress_messages,
                    "max_output_tokens": 200  # Small response for summary
                }
            else:
                payload = {
                    "model": model,
                    "messages": compress_messages,
                    "temperature": 0.1,
                    "max_tokens": 200,  # Small response for summary
                    "stream": False  # Always explicitly disable streaming
                }
            
            # Compression should use instruction model, not reasoning model
            # Don't enable reasoning for compression (it's just for summarization)
            
            response = self.session.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            response_json = response.json()
            if self.config.llm_provider == "openai":
                summary = self._extract_responses_text(response_json)
            else:
                summary = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Reconstruct compressed message history
            compressed = []
            if system_msg:
                compressed.append(system_msg)
            
            # Add summary as a single user message
            if summary:
                compressed.append({
                    "role": "user",
                    "content": f"[Previous conversation summary: {summary}]"
                })
            elif summary_parts:
                compressed.append({
                    "role": "user",
                    "content": f"[Previous conversation: {'; '.join(summary_parts)}]"
                })
            
            # Add recent messages
            compressed.extend(recent_messages)
            
            logger.info(f"Compressed {len(messages)} messages to {len(compressed)} messages")
            return compressed
            
        except Exception as e:
            logger.warning(f"Failed to compress messages with LLM: {e}. Using fallback: keeping only recent messages.")
            # Fallback: just keep system + recent messages with a simple note
            compressed = []
            if system_msg:
                compressed.append(system_msg)
            if summary_parts:
                compressed.append({
                    "role": "user",
                    "content": f"[Previous steps: {'; '.join(summary_parts)}]"
                })
            # Ensure recent_messages is a list
            if recent_messages:
                compressed.extend(recent_messages)
            elif not compressed:
                # If we have nothing, at least keep the system message or return minimal
                if system_msg:
                    compressed = [system_msg]
                else:
                    # Last resort: return at least the last message
                    compressed = messages[-1:] if messages else []
            
            logger.info(f"Fallback compression: {len(messages)} messages to {len(compressed)} messages")
            return compressed if compressed else messages[-max_keep:] if messages else []
    
    def _call_llm(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]], tool_choice: str = "auto", retry_on_token_error: bool = True, use_query_model: bool = False) -> Dict[str, Any]:
        """
        Call the LLM via OpenAI-compatible API (local vLLM or OpenRouter)
        
        Args:
            messages: Conversation messages
            tools: Available tools (if any)
            tool_choice: Tool choice mode
            retry_on_token_error: Whether to retry on token errors
            use_query_model: If True, use query generation model (reasoning). If False, use tool call model (instruction).
        """
        
        # Determine API endpoint and headers based on provider
        if self.config.llm_provider == "olmo":
            # OpenRouter API
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.config.openrouter_api_key}",
                "HTTP-Referer": "https://github.com/matra-research",  # Optional: for analytics
                "X-Title": "MatRA Research Agent",  # Optional: for analytics
                "Content-Type": "application/json",  # Explicit content type
                "Accept": "application/json"  # Explicit accept header (NOT text/event-stream for streaming)
            }
            # Use query model for query generation, tool model for tool calls
            if use_query_model and self.config.openrouter_query_model:
                model = self.config.openrouter_query_model
                # Reasoning models are only for query generation
                use_reasoning = "think" in model.lower()
            else:
                model = self.config.openrouter_model
                # Tool calls should NOT use reasoning models - use instruction models only
                use_reasoning = False
                if "think" in model.lower():
                    logger.warning(f"Warning: Using reasoning model '{model}' for tool calls. This is not recommended. Use an instruction-tuned model like 'olmo-3-7b-instruct' instead.")
        elif self.config.llm_provider == "openai":
            base_url = self._normalize_openai_base_url()
            url = f"{base_url}/responses"
            headers = {
                "Authorization": f"Bearer {self.config.openai_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            if use_query_model and self.config.openai_query_model:
                model = self.config.openai_query_model
            else:
                model = self.config.openai_model
            use_reasoning = False
        else:
            # Local vLLM
            url = f"{self.config.vllm_url}/v1/chat/completions"
            headers = {}
            model = self.config.model_name
            use_reasoning = False
        
        # Prepare messages - preserve reasoning_details if present (required for Olmo models)
        prepared_messages = []
        for msg in messages:
            prepared_msg = {k: v for k, v in msg.items() if k != "reasoning_details"}
            # Preserve reasoning_details if present (OpenRouter requirement for continuing conversations)
            if "reasoning_details" in msg and use_reasoning:
                prepared_msg["reasoning_details"] = msg["reasoning_details"]
            prepared_messages.append(prepared_msg)
        
        # CRITICAL: Build payload carefully to ensure streaming is disabled
        # Some OpenRouter providers (like ModelRun) don't support tools in streaming mode
        # We MUST set stream=False as a boolean (not string) and ensure it's in the payload
        if self.config.llm_provider == "openai":
            # Responses API: gpt-5-nano and some models do not support temperature
            payload = {
                "model": model,
                "input": prepared_messages,
                "max_output_tokens": 2000
            }
        else:
            payload = {
                "model": model,
                "messages": prepared_messages,
                "temperature": 0.1,
                "max_tokens": 2000
            }
        
        # Enable reasoning for Olmo models (OpenRouter feature)
        # reasoning must be an object, not a boolean
        if use_reasoning:
            payload["reasoning"] = {
                "enabled": True  # Enable reasoning with default parameters
                # Alternative options:
                # "effort": "high",  # "high", "medium", "low", "minimal", "none"
                # "max_tokens": 2000,  # For models that support direct token allocation
            }
        
        # Only include tools and tool_choice if tools are provided
        if tools:
            # Responses API expects top-level name/description/parameters; Chat Completions uses nested "function"
            payload["tools"] = self._tools_for_responses_api(tools) if self.config.llm_provider == "openai" else tools
            payload["tool_choice"] = tool_choice
        if self.config.llm_provider != "openai":
            # CRITICAL: Explicitly set stream=False when tools are present
            # Use boolean False, not string "false"
            payload["stream"] = False
        
        # Don't set tool_choice to "none" if no tools - some servers don't support it
        
        # Log full payload details for debugging (but don't log API key)
        payload_log = {k: v for k, v in payload.items() if k != "messages" or len(str(v)) < 100}
        logger.info(f"Making LLM call. Model: {model}, Stream: {payload.get('stream')} (type: {type(payload.get('stream'))}), Tools: {bool(tools)}, Payload keys: {list(payload.keys())}")
        
        try:
            # Log the actual request being sent (sanitized)
            import json as json_module
            payload_str = json_module.dumps(payload, indent=2)
            # Truncate messages for logging
            if "messages" in payload:
                payload_for_log = payload.copy()
                payload_for_log["messages"] = [f"<message {i} ({len(str(m))} chars)>" for i, m in enumerate(payload["messages"])]
                logger.debug(f"Request payload (sanitized): {json_module.dumps(payload_for_log, indent=2)}")
            
            # Make the request
            response = self.session.post(url, json=payload, headers=headers, timeout=120)  # Longer timeout for OpenRouter
            response.raise_for_status()
            result = response.json()
            if self.config.llm_provider == "openai":
                tool_calls = []
                for item in result.get("output", []):
                    if item.get("type") == "function_call":
                        tool_calls.append({
                            "id": item.get("call_id") or item.get("id"),
                            "type": "function",
                            "function": {
                                "name": item.get("name"),
                                "arguments": item.get("arguments", "{}")
                            }
                        })
                normalized = {
                    "choices": [
                        {
                            "message": {
                                "content": self._extract_responses_text(result),
                                "tool_calls": tool_calls or []
                            }
                        }
                    ]
                }
                return normalized
            
            # Extract reasoning details if available (OpenRouter feature)
            # reasoning_details is in choices[].message.reasoning_details, not at top level
            if use_reasoning:
                choice = result.get("choices", [{}])[0]
                message = choice.get("message", {})
                reasoning_details = message.get("reasoning_details", [])
                if reasoning_details:
                    logger.info(f"Received {len(reasoning_details)} reasoning detail objects")
                    # Log first reasoning detail for debugging
                    if len(reasoning_details) > 0 and isinstance(reasoning_details[0], dict):
                        first_detail = reasoning_details[0]
                        detail_type = first_detail.get("type", "unknown")
                        if detail_type == "reasoning.text":
                            text = first_detail.get("text", "")[:200]
                            logger.debug(f"First reasoning text: {text}...")
                        elif detail_type == "reasoning.summary":
                            summary = first_detail.get("summary", "")[:200]
                            logger.debug(f"First reasoning summary: {summary}...")
            
            return result
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_detail = response.json()
            except:
                error_detail = response.text
            
            error_message = str(error_detail)
            
            # Check if it's a streaming mode error (tools are not supported in streaming)
            is_streaming_error = "streaming mode" in error_message.lower() or "tools are not supported" in error_message.lower()
            if is_streaming_error and tools:
                logger.error(f"Streaming error detected despite stream=False. Full error: {error_detail}")
                logger.error(f"Payload stream value: {payload.get('stream')}, type: {type(payload.get('stream'))}")
                logger.error(f"Model: {model}, Provider appears to be ModelRun based on error")
                
                # ModelRun provider seems to have a bug where it ignores stream=False
                # Try omitting stream parameter entirely (some APIs interpret absence as non-streaming)
                retry_payload = {
                    "model": model,
                    "messages": prepared_messages,
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "tools": tools,
                    "tool_choice": tool_choice
                    # Intentionally NOT including stream parameter - let API default to non-streaming
                }
                
                # Add reasoning if it was in original
                if use_reasoning and "reasoning" in payload:
                    retry_payload["reasoning"] = payload["reasoning"]
                
                logger.warning(f"Retrying WITHOUT stream parameter (omitted entirely). Some providers default to non-streaming when parameter is absent.")
                logger.info(f"Retry payload keys: {list(retry_payload.keys())}, stream parameter: {'present' if 'stream' in retry_payload else 'OMITTED'}")
                
                try:
                    response = self.session.post(url, json=retry_payload, headers=headers, timeout=120)
                    response.raise_for_status()
                    logger.info("✅ Retry without stream parameter succeeded!")
                    return response.json()
                except Exception as retry_error:
                    logger.error(f"Retry without stream parameter also failed: {retry_error}")
                    # Try one more time with stream explicitly set to False again
                    retry_payload["stream"] = False
                    logger.warning("Trying one more time with stream=False explicitly set...")
                    try:
                        response = self.session.post(url, json=retry_payload, headers=headers, timeout=120)
                        response.raise_for_status()
                        logger.info("✅ Retry with stream=False succeeded!")
                        return response.json()
                    except Exception as final_error:
                        logger.error(f"All retry attempts failed. This appears to be a ModelRun provider limitation.")
                        logger.error(f"Original error: {e}")
                        logger.error(f"Response: {error_detail}")
                        # Provide helpful error message
                        raise Exception(
                            f"ModelRun provider via OpenRouter does not support tools with this model. "
                            f"Error: {error_detail}. "
                            f"Try using a different model that supports tools, or contact OpenRouter support. "
                            f"Suggested models: meta-llama/llama-3.3-70b-instruct, anthropic/claude-3.5-sonnet"
                        )
            
            # Check if it's a token limit error
            is_token_error = "maximum context length" in error_message or "requested" in error_message and "tokens" in error_message
            
            if is_token_error and retry_on_token_error:
                logger.warning(f"Token limit exceeded. Attempting to compress conversation history...")
                try:
                    # Ensure messages is valid before compression
                    if not messages:
                        raise ValueError("Cannot compress: messages is None or empty")
                    
                    # Compress messages and retry
                    compressed_messages = self._compress_messages(messages)
                    
                    # Validate compressed messages
                    if not compressed_messages:
                        logger.error("Compression returned empty list. Using fallback: keeping only system message and last 3 messages.")
                        # Fallback: keep only system + last 3 messages
                        if messages and len(messages) > 0:
                            system_msg = messages[0] if messages[0].get("role") == "system" else None
                            compressed_messages = []
                            if system_msg:
                                compressed_messages.append(system_msg)
                            compressed_messages.extend(messages[-3:])
                        else:
                            raise ValueError("Cannot create fallback: no messages available")
                    
                    # Reduce max_tokens aggressively to leave room for response
                    # If we're hitting limits, we need to be conservative
                    if self.config.llm_provider == "openai":
                        payload["input"] = compressed_messages
                        payload["max_output_tokens"] = min(1000, payload.get("max_output_tokens", 1000))
                    else:
                        payload["messages"] = compressed_messages
                        payload["max_tokens"] = min(1000, payload["max_tokens"])  # Reduce max_tokens more aggressively
                        # Ensure stream is explicitly False (already set, but double-check)
                        payload["stream"] = False
                    
                    logger.info(f"Retrying with compressed messages ({len(compressed_messages)} messages, reduced from {len(messages)})")
                    response = self.session.post(url, json=payload, timeout=60)
                    response.raise_for_status()
                    if self.config.llm_provider == "openai":
                        result = response.json()
                        tool_calls = []
                        for item in result.get("output", []):
                            if item.get("type") == "function_call":
                                tool_calls.append({
                                    "id": item.get("call_id") or item.get("id"),
                                    "type": "function",
                                    "function": {
                                        "name": item.get("name"),
                                        "arguments": item.get("arguments", "{}")
                                    }
                                })
                        return {
                            "choices": [
                                {
                                    "message": {
                                        "content": self._extract_responses_text(result),
                                        "tool_calls": tool_calls or []
                                    }
                                }
                            ]
                        }
                    return response.json()
                except Exception as retry_error:
                    logger.error(f"Retry after compression failed: {retry_error}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Fall back to original error
                    logger.error(f"LLM API call failed: {e}")
                    logger.error(f"Response: {error_detail}")
                    raise
            
            logger.error(f"LLM API call failed: {e}")
            logger.error(f"Response: {error_detail}")
            raise
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def _call_openai_responses(
        self,
        instructions: str,
        input_list: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        max_output_tokens: int = 2000,
    ) -> Dict[str, Any]:
        """Call OpenAI Responses API with input list (no Chat-format messages). Returns raw response."""
        base_url = self._normalize_openai_base_url()
        url = f"{base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self.config.openai_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        model = self.config.openai_model
        payload = {
            "model": model,
            "instructions": instructions,
            "input": input_list,
            "max_output_tokens": max_output_tokens,
        }
        if tools:
            payload["tools"] = self._tools_for_responses_api(tools)
            payload["tool_choice"] = tool_choice
        logger.info(f"Making OpenAI Responses call. Model: {model}, Tools: {bool(tools)}, input items: {len(input_list)}")
        response = self.session.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        return response.json()

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

    def _execute_workflow_openai(
        self,
        system_prompt: str,
        user_input: str,
        tools: List[Dict[str, Any]],
        max_iterations: int = 20,
    ) -> Dict[str, Any]:
        """Execute workflow using OpenAI Responses API input format (instructions + input list with output and function_call_output)."""
        REQUIRED_WORKFLOW_TOOLS = [
            "crossref_search",
            "combine_and_deduplicate",
            "download_articles",
            "build_database",
            "organize_xmls",
        ]
        openai_input_list: List[Dict[str, Any]] = [{"role": "user", "content": user_input}]
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Agent iteration {iteration}/{max_iterations} (OpenAI Responses API)")
            try:
                result = self._call_openai_responses(
                    instructions=system_prompt,
                    input_list=openai_input_list,
                    tools=tools,
                    tool_choice="auto",
                    max_output_tokens=2000,
                )
            except Exception as e:
                error_msg = f"LLM call failed: {str(e)}"
                logger.error(error_msg)
                self.state.errors.append(error_msg)
                return {
                    "success": False,
                    "summary": "",
                    "tool_results": self.state.tool_results,
                    "errors": self.state.errors,
                    "reasoning_history": self.state.reasoning_history,
                }
            output_items = result.get("output", [])
            tool_calls = []
            for item in output_items:
                if item.get("type") == "function_call":
                    tool_calls.append({
                        "id": item.get("call_id") or item.get("id"),
                        "type": "function",
                        "function": {
                            "name": item.get("name"),
                            "arguments": item.get("arguments", "{}"),
                        },
                    })
            if len(tool_calls) > 1:
                logger.warning(f"Model returned {len(tool_calls)} tool calls; processing first only.")
                tool_calls = [tool_calls[0]]
            openai_input_list = list(openai_input_list)
            openai_input_list.extend(output_items)
            if not tool_calls:
                completed = [tr["tool"] for tr in self.state.tool_results]
                remaining = [t for t in REQUIRED_WORKFLOW_TOOLS if t not in completed]
                if remaining:
                    next_tool = remaining[0]
                    logger.info(f"No tool call in response; reminding agent to run remaining steps: {remaining}")
                    openai_input_list.append({
                        "role": "user",
                        "content": f"The workflow is not complete. You must still run these tools in order: {', '.join(remaining)}. Call the next tool now: {next_tool}. Use the corpus_dir, output_csv, and other paths from the user's instructions above.",
                    })
                    continue
                openai_input_list.append({
                    "role": "user",
                    "content": "Workflow completed. Provide a summary of what was accomplished.",
                })
                try:
                    final_result = self._call_openai_responses(
                        instructions=system_prompt,
                        input_list=openai_input_list,
                        tools=[],
                        max_output_tokens=1000,
                    )
                    final_text = self._extract_responses_text(final_result)
                except Exception as e:
                    logger.warning(f"Final summary call failed: {e}")
                    final_text = "Workflow completed."
                return {
                    "success": True,
                    "summary": final_text,
                    "tool_results": self.state.tool_results,
                    "errors": self.state.errors,
                    "reasoning_history": self.state.reasoning_history,
                }
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "?")
                step_num = next((i + 1 for i, t in enumerate(REQUIRED_WORKFLOW_TOOLS) if t == tool_name), None)
                step_label = f"Step {step_num}/5: {tool_name}" if step_num else tool_name
                logger.info(f"=== {step_label} ===")
                exec_result = self._process_tool_call(tool_call)
                call_id = tool_call.get("id")
                status = "OK" if exec_result.get("success") else "FAILED"
                summary = ""
                if isinstance(exec_result.get("result"), dict):
                    r = exec_result["result"]
                    if "total_articles" in r:
                        summary = f" ({r.get('total_articles', 0)} articles)"
                    elif "unique_count" in r:
                        summary = f" ({r.get('unique_count', 0)} unique DOIs)"
                    elif "successful" in r:
                        summary = f" ({r.get('successful', 0)} downloaded)"
                logger.info(f"=== {step_label} -> {status}{summary} ===")
                openai_input_list.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps({
                        "success": exec_result["success"],
                        "result": exec_result["result"],
                        "error": exec_result["error"],
                    }),
                })
        return {
            "success": False,
            "summary": "Maximum iterations reached",
            "tool_results": self.state.tool_results,
            "errors": self.state.errors + ["Maximum iterations reached"],
            "reasoning_history": self.state.reasoning_history,
        }

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
You have access to the following tools and MUST call them in this exact order:
1. crossref_search - Search CrossRef API for papers
2. combine_and_deduplicate - Combine search results and remove duplicates
3. download_articles - Download article XMLs from DOIs
4. build_database - Extract metadata from downloaded XMLs (corpus_dir and output_csv from user instructions)
5. organize_xmls - Organize XMLs into combined directory (corpus_dir and database_csv from user instructions)

You MUST execute all five tools in sequence. Do not stop after download_articles; you must also call build_database then organize_xmls.
IMPORTANT: You can only make ONE tool call at a time. After each tool call completes, you will receive the result and must then make the next tool call.
Always use the paths (output_dir, corpus_dir, consolidated CSV, database CSV) provided in the user's instructions when calling tools."""

        tools = self.registry.get_all_schemas()

        # OpenAI Responses API: use input list (instructions + input), append response output and function_call_output; no Chat-format tool_calls
        if self.config.llm_provider == "openai":
            return self._execute_workflow_openai(
                system_prompt=system_prompt,
                user_input=user_input,
                tools=tools,
                max_iterations=max_iterations,
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Agent iteration {iteration}/{max_iterations}")
            
            # Call LLM (with automatic retry on token errors)
            try:
                response = self._call_llm(messages, tools, retry_on_token_error=True)
            except Exception as e:
                error_msg = f"LLM call failed: {str(e)}"
                logger.error(error_msg)
                # Try one more time with compressed messages if it's a token error
                if "maximum context length" in str(e) or "tokens" in str(e):
                    try:
                        logger.info("Attempting to compress conversation history and retry...")
                        if not messages:
                            raise ValueError("Cannot compress: messages is None or empty")
                        
                        compressed = self._compress_messages(messages)
                        
                        # Validate compressed messages
                        if not compressed:
                            logger.error("Compression returned empty. Using minimal fallback.")
                            if messages:
                                system_msg = messages[0] if messages[0].get("role") == "system" else None
                                compressed = []
                                if system_msg:
                                    compressed.append(system_msg)
                                compressed.extend(messages[-2:])  # Keep last 2 messages
                            else:
                                raise ValueError("No messages available for compression")
                        
                        response = self._call_llm(compressed, tools, retry_on_token_error=False)
                        messages = compressed  # Update to use compressed version going forward
                        logger.info(f"Successfully compressed and retried. Continuing with compressed history ({len(compressed)} messages).")
                    except Exception as retry_error:
                        logger.error(f"Retry after compression failed: {retry_error}")
                        import traceback
                        logger.error(traceback.format_exc())
                        self.state.errors.append(error_msg)
                        break
                else:
                    self.state.errors.append(error_msg)
                    break
            
            # Extract response
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])
            
            # Extract reasoning details if available (OpenRouter Olmo models)
            # reasoning_details is in choices[].message.reasoning_details, not at top level
            reasoning_details = message.get("reasoning_details", [])
            if reasoning_details:
                logger.info(f"Received {len(reasoning_details)} reasoning detail objects from model")
                # Store reasoning in state for potential display
                self.state.reasoning_history.append({
                    "iteration": iteration,
                    "reasoning_steps": reasoning_details
                })
                # Log first reasoning detail for debugging
                if len(reasoning_details) > 0 and isinstance(reasoning_details[0], dict):
                    first_detail = reasoning_details[0]
                    detail_type = first_detail.get("type", "unknown")
                    if detail_type == "reasoning.text":
                        text = first_detail.get("text", "")[:200]
                        logger.debug(f"First reasoning text: {text}...")
                    elif detail_type == "reasoning.summary":
                        summary = first_detail.get("summary", "")[:200]
                        logger.debug(f"First reasoning summary: {summary}...")
            
            # If multiple tool calls, only process the first one (model limitation)
            if len(tool_calls) > 1:
                logger.warning(f"Model returned {len(tool_calls)} tool calls, but only supports one at a time. Processing first tool call only.")
                tool_calls = [tool_calls[0]]
            
            # Add assistant message to history (with only first tool call if multiple)
            assistant_msg = {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls if tool_calls else None
            }
            
            # Preserve reasoning details for Olmo models (OpenRouter requirement)
            # reasoning_details must be preserved exactly as received for continuing conversations
            if reasoning_details and self.config.llm_provider == "olmo":
                assistant_msg["reasoning_details"] = reasoning_details
                logger.debug(f"Preserving {len(reasoning_details)} reasoning detail objects in message history")
            
            messages.append(assistant_msg)
            
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
                    "errors": self.state.errors,
                    "reasoning_history": self.state.reasoning_history
                }
            
            # Process tool calls (should only be one now)
            REQUIRED_ORDER = ["crossref_search", "combine_and_deduplicate", "download_articles", "build_database", "organize_xmls"]
            tool_messages = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "?")
                step_num = next((i + 1 for i, t in enumerate(REQUIRED_ORDER) if t == tool_name), None)
                step_label = f"Step {step_num}/5: {tool_name}" if step_num else tool_name
                logger.info(f"=== {step_label} ===")
                result = self._process_tool_call(tool_call)
                status = "OK" if result.get("success") else "FAILED"
                summary = ""
                if isinstance(result.get("result"), dict):
                    r = result["result"]
                    if "total_articles" in r:
                        summary = f" ({r.get('total_articles', 0)} articles)"
                    elif "unique_count" in r:
                        summary = f" ({r.get('unique_count', 0)} unique DOIs)"
                    elif "successful" in r:
                        summary = f" ({r.get('successful', 0)} downloaded)"
                logger.info(f"=== {step_label} -> {status}{summary} ===")
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
            "errors": self.state.errors + ["Maximum iterations reached"],
            "reasoning_history": self.state.reasoning_history
        }
    
    def reset_state(self):
        """Reset agent state"""
        self.state = AgentState()
    
    def generate_search_queries(self, research_description: str, num_queries: int = 5) -> List[str]:
        """
        Generate search queries from natural language research description
        
        Args:
            research_description: Natural language description of research area
            num_queries: Number of queries to generate (default: 5)
            
        Returns:
            List of search query strings
        """
        system_prompt = f"Generate exactly {num_queries} search queries (one per line) from the research description. Each query should not have more than 4 words."

        try:
            user_message = f"{research_description}\n\nGenerate exactly {num_queries} search queries. Return only the queries, one per line, no JSON, no formatting."
            # Use query model (reasoning) for query generation
            response = self._call_llm(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}], 
                tools=[],
                use_query_model=True  # Use reasoning model for query generation
            )
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            # Clean up content - remove any JSON-like structures or tool call artifacts
            if content.startswith('{') or '"name"' in content or '"parameters"' in content:
                logger.warning("LLM returned tool call format, attempting to extract queries")
                # Try to extract queries from JSON structure if present
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "parameters" in parsed:
                        # Extract from parameters if available
                        params = parsed.get("parameters", {})
                        if "research_area" in params:
                            content = params["research_area"]
                        elif "queries" in params:
                            content = "\n".join(params["queries"]) if isinstance(params["queries"], list) else str(params["queries"])
                except (json.JSONDecodeError, KeyError):
                    # If JSON parsing fails, try to extract from lines
                    lines = content.split('\n')
                    content = '\n'.join([line for line in lines if not (line.strip().startswith('{') or '"name"' in line or '"parameters"' in line)])
            
            # Parse queries from newline-separated content
            queries = [q.strip() for q in content.split('\n') if q.strip() and not q.strip().startswith('{')]
            if not queries:
                logger.warning("No queries generated, using fallback")
                # Fallback: use first few words of description
                words = research_description.split()[:5]
                queries = [' '.join(words)]
            
            logger.info(f"Generated {len(queries)} search queries: {queries}")
            return queries
            
        except Exception as e:
            logger.error(f"Error generating queries: {e}")
            # Fallback: use first few words of description
            words = research_description.split()[:5]
            return [' '.join(words)]