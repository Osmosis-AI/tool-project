#!/usr/bin/env python3

import asyncio
import json
import os
import hashlib
import time
import logging
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.logging import RichHandler
from pathlib import Path
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# LiteLLM imports
import litellm
from litellm import acompletion

# MCP imports (keeping the MCP functionality)
from langchain_mcp_adapters.client import MultiServerMCPClient

from experiment.experiment_control import test_batteries, retry_count
from env import OPENROUTER_KEY

# Token counting imports
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None

console = Console()

# Configure litellm
litellm.set_verbose = True
litellm.api_base = "https://openrouter.ai/api/v1"

# Add custom model pricing to avoid cost calculation errors
def setup_custom_model_pricing():
    """Add custom pricing for models not in LiteLLM's database"""
    custom_pricing = {
        "moonshotai/kimi-k2": {
            "input_cost_per_token": 0.00000014,  # $0.14/M input tokens
            "output_cost_per_token": 0.00000249,  # $2.49/M output tokens
            "max_tokens": 32000
        },
        # Add other problematic models here
        "anthropic/claude-sonnet-4": {
            "input_cost_per_token": 0.000003,
            "output_cost_per_token": 0.000015,
            "max_tokens": 200000
        },
        "openrouter/openai/o3": {
            "input_cost_per_token": 0.0000002,
            "output_cost_per_token": 0.0000008,
            "max_tokens": 128000
        },
        "openai/gpt-oss-120b": {
            "input_cost_per_token": 0.00000015,
            "output_cost_per_token": 0.0000006,
            "max_tokens": 128000
        },
        "openai/gpt-5": {
            "input_cost_per_token": 0.000000125, # $1.25
            "output_cost_per_token": 0.000001, # $10
            "max_tokens": 128000
        }
    }
    
    # Register custom pricing with LiteLLM
    for model_name, pricing in custom_pricing.items():
        try:
            litellm.model_cost[model_name] = pricing
        except Exception:
            pass  # Ignore if registration fails

setup_custom_model_pricing()

# Alternative: Suppress cost calculation errors entirely
litellm.suppress_debug_info = True

@dataclass
class RunResult:
    query: str
    model: str
    experiment: str
    timestamp: str
    query_id: str
    run_hash: str
    success: bool
    response: Optional[str]
    error: Optional[str]
    metrics: Dict[str, Any]


@dataclass
class ModelSummary:
    success_rate: float
    successful_runs: int
    total_runs: int
    avg_latency: float
    avg_accuracy: float
    tools_used: float
    input_tokens: float
    output_tokens: float
    total_tokens: float
    avg_cost: float


# Simplified MCP server info class for compatibility
class MCPServerInfo:
    def __init__(
        self,
        name: str,
        connected: bool,
        tools: List[str] = None,
        resources: int = 0,
        prompts: int = 0,
    ):
        self.name = name
        self.connected = connected
        self.tools = tools or []
        self.resources = resources
        self.prompts = prompts

    def to_dict(self) -> Dict[str, Any]:
        """Convert MCP server info to dictionary for serialization"""
        return {
            "name": self.name,
            "connected": self.connected,
            "tools": self.tools,
            "resources": self.resources,
            "prompts": self.prompts,
        }


class ExperimentRunner:
    def __init__(self, output_dir: Optional[str] = None, max_concurrent_requests: int = 5):
        self.mcp_client: Optional[MultiServerMCPClient] = None
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Add semaphore for controlling concurrent requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        # Setup logging with more verbose configuration
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[RichHandler(console=console, rich_tracebacks=True)],
        )
        self.logger = logging.getLogger("experiment_runner")
        
        # Set LiteLLM logger to DEBUG
        logging.getLogger("litellm").setLevel(logging.DEBUG)

    def convert_mcp_config_to_langchain(
        self, mcp_server_configs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Convert experiment_control.py MCP configs to LangChain format"""
        langchain_configs = {}

        for name, config in mcp_server_configs.items():
            if config.get("type") == "http":
                # HTTP server config
                langchain_configs[name] = {
                    "url": config.get("url"),
                    "transport": "streamable_http",
                    "headers": config.get("headers", {}),
                }
            else:
                # Stdio server config
                langchain_configs[name] = {
                    "command": config.get("command"),
                    "args": config.get("args", []),
                    "transport": "stdio",
                    "env": dict(os.environ, **config.get("env", {})),
                }

        return langchain_configs

    async def setup_mcp_client(
        self, mcp_server_configs: Dict[str, Dict[str, Any]]
    ) -> Optional[MultiServerMCPClient]:
        """Setup LangChain MCP client with robust error handling"""
        console.print(
            f"[blue]Setting up LangChain MCP client with {len(mcp_server_configs)} servers...[/blue]"
        )

        if not mcp_server_configs:
            console.print("[yellow]No MCP server configurations provided[/yellow]")
            return None

        try:
            # Convert configs to LangChain format
            console.print("[blue]Converting MCP configs to LangChain format...[/blue]")
            langchain_configs = self.convert_mcp_config_to_langchain(mcp_server_configs)
            console.print(f"[cyan]Converted {len(langchain_configs)} server configurations[/cyan]")
            
            # Print detailed config for debugging
            for name, config in list(langchain_configs.items())[:3]:  # Show first 3 configs
                console.print(f"[cyan]  {name}: {config}[/cyan]")

            # Create and return MCP client
            console.print("[blue]Creating MultiServerMCPClient...[/blue]")
            client = MultiServerMCPClient(langchain_configs)

            console.print(
                f"[green]LangChain MCP client initialized with {len(langchain_configs)} server configurations[/green]"
            )
            
            # Test the client by trying to get tools immediately
            console.print("[blue]Testing MCP client by fetching tools...[/blue]")
            try:
                test_tools = await asyncio.wait_for(client.get_tools(), timeout=30.0)
                console.print(f"[green]Successfully retrieved {len(test_tools)} tools from MCP client[/green]")
                return client
            except asyncio.TimeoutError:
                console.print("[red]MCP client tool retrieval timed out during setup[/red]")
                return None
            except Exception as e:
                console.print(f"[red]Failed to retrieve tools during MCP client setup: {e}[/red]")
                import traceback
                console.print(f"[red]Tool retrieval traceback: {traceback.format_exc()}[/red]")
                return None
                
        except Exception as e:
            console.print(f"[red]Failed to initialize MCP client: {e}[/red]")
            import traceback
            console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
            return None

    async def cleanup_mcp_client(self, client: Optional[MultiServerMCPClient]):
        """Clean up MCP client with better resource management"""
        if client:
            console.print("[blue]Cleaning up MCP client...[/blue]")
            try:
                # Give any background tasks time to complete
                await asyncio.sleep(0.1)
                
                # Try multiple cleanup methods to ensure proper resource cleanup
                if hasattr(client, 'close'):
                    await client.close()
                elif hasattr(client, '__aexit__'):
                    await client.__aexit__(None, None, None)
                elif hasattr(client, 'cleanup'):
                    await client.cleanup()
                    
                # Force garbage collection to help with resource cleanup
                import gc
                gc.collect()
                
            except Exception as e:
                console.print(f"[yellow]Warning during MCP cleanup: {e}[/yellow]")
        else:
            console.print("[blue]No MCP client to clean up[/blue]")

    def generate_query_id(self, query: str, model: str, timestamp: str) -> str:
        """Generate a unique query ID"""
        content = f"{query}{model}{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def generate_run_hash(
        self, query: str, model: str, experiment: str, timestamp: str
    ) -> str:
        """Generate a unique run hash"""
        content = f"{query}{model}{experiment}{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def setup_run_logging(self, query_id: str) -> tuple[logging.Logger, str]:
        """Setup per-run logging with comprehensive capture"""
        log_file = (
            self.output_dir
            / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{query_id}.log"
        )

        # Create a new logger for this run
        run_logger = logging.getLogger(f"run_{query_id}")
        run_logger.setLevel(logging.DEBUG)
        run_logger.propagate = False

        # Remove existing handlers
        for handler in run_logger.handlers[:]:
            run_logger.removeHandler(handler)

        # Add file handler with detailed formatting
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(detailed_formatter)
        file_handler.setLevel(logging.DEBUG)
        run_logger.addHandler(file_handler)
        
        # Also capture LiteLLM logs for this run
        litellm_logger = logging.getLogger("litellm")
        litellm_logger.addHandler(file_handler)
        litellm_logger.setLevel(logging.DEBUG)

        return run_logger, str(log_file)

    async def query_model(
        self,
        query: str,
        model: str,
        tools: List[Any],
        mcp_client: Optional[MultiServerMCPClient] = None,
        run_logger: Optional[logging.Logger] = None,
    ) -> Dict[str, Any]:
        """Query a model using LiteLLM with pre-cached MCP tools"""
        start_time = time.time()
        
        if run_logger:
            run_logger.info(f"Starting model query: {model}")
            run_logger.info(f"Query: {query}")
            run_logger.info(f"Available tools: {len(tools)}")
            if tools:
                tool_names = [getattr(tool, 'name', str(tool)) for tool in tools]
                run_logger.info(f"Tool names: {tool_names}")

        try:
            # Convert model name for LiteLLM with OpenRouter
            if model.startswith("openrouter/"):
                litellm_model = model
            elif model == "openai/o3":
                # Special handling for O3 model
                litellm_model = "openrouter/openai/o3"
            else:
                # For OpenRouter, prefix with openrouter/
                litellm_model = f"openrouter/{model}"

            # Set up messages
            messages = [
                {"role": "system", "content": "You are an AI assistant."},
                {"role": "user", "content": query}
            ]

            # Add tool context if tools are available
            if tools and self._model_supports_functions(model):
                system_content = (
                    f"You are an AI assistant with access to MCP (Model Context Protocol) servers and their tools. "
                    f"You have access to {len(tools)} tools (some may be prefixed with server names to ensure uniqueness). "
                    f"Use these tools to help answer questions and complete tasks. If one fails, try using other tools to achieve the same outcome."
                )
                
                # Add "Reasoning: high" for the gpt-oss-120b model
                if model == "openai/gpt-oss-120b":
                    system_content += "\n\nReasoning: high"
                
                messages[0]["content"] = system_content
                
                if run_logger:
                    run_logger.info(f"Using model with tools support")
                    run_logger.info(f"Tool names available: {[getattr(tool, 'name', str(tool)) for tool in tools]}")

            if run_logger:
                run_logger.info(f"Making LiteLLM call to model: {litellm_model}")
                run_logger.info(f"Messages: {messages}")

            # Make the LiteLLM call
            call_params = {
                "model": litellm_model,
                "messages": messages,
                "api_key": OPENROUTER_KEY,
                "api_base": "https://openrouter.ai/api/v1",
                "temperature": 0.7,
                "stream": False
            }
            
            # Add tools if available and model supports them
            if tools and self._model_supports_functions(model):
                # Encourage the model to call tools when useful
                call_params["tool_choice"] = "auto"
                # Convert MCP tools to OpenAI function format
                openai_tools = []
                
                if run_logger:
                    run_logger.info(f"Starting tool conversion for {len(tools)} MCP tools")
                
                for i, tool in enumerate(tools):
                    if run_logger:
                        run_logger.info(f"Processing tool {i}: {tool} (type: {type(tool)})")
                        run_logger.info(f"Tool attributes: name={getattr(tool, 'name', 'MISSING')}, description={getattr(tool, 'description', 'MISSING')}")
                    
                    # Check if tool has required attributes
                    if not hasattr(tool, 'name'):
                        if run_logger:
                            run_logger.warning(f"Tool {i} missing 'name' attribute: {tool}")
                        continue
                    
                    if not hasattr(tool, 'description'):
                        if run_logger:
                            run_logger.warning(f"Tool {i} missing 'description' attribute: {tool}")
                        continue
                    
                    tool_name = tool.name
                    if not tool_name or not str(tool_name).strip():
                        if run_logger:
                            run_logger.warning(f"Tool {i} has empty or None name: '{tool_name}'")
                        continue
                    
                    tool_name = str(tool_name).strip()
                    tool_description = tool.description or "No description available"
                    
                    tool_spec = {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": str(tool_description)
                        }
                    }
                    
                    # Add parameters if available
                    if hasattr(tool, 'args_schema') and tool.args_schema:
                        try:
                            schema = None
                            if hasattr(tool.args_schema, 'model_json_schema'):
                                # Pydantic v2 model
                                schema = tool.args_schema.model_json_schema()
                            elif hasattr(tool.args_schema, 'schema'):
                                # Pydantic v1 model or other schema object
                                schema = tool.args_schema.schema()
                            elif isinstance(tool.args_schema, dict):
                                # Already a dictionary schema - use it directly
                                schema = tool.args_schema.copy()
                            else:
                                # Try to convert to dict if it has dict-like behavior
                                schema = dict(tool.args_schema)
                            
                            if schema and isinstance(schema, dict):
                                # Clean up the schema for OpenAI function format
                                clean_schema = {
                                    "type": schema.get("type", "object"),
                                    "properties": schema.get("properties", {}),
                                    "required": schema.get("required", [])
                                }
                                tool_spec["function"]["parameters"] = clean_schema
                                if run_logger:
                                    run_logger.info(f"Added parameters for tool {tool_name}: {len(clean_schema.get('properties', {}))} properties")
                            else:
                                # Fallback: provide minimal parameters structure
                                tool_spec["function"]["parameters"] = {
                                    "type": "object",
                                    "properties": {},
                                    "required": []
                                }
                                if run_logger:
                                    run_logger.warning(f"Schema extraction returned non-dict for tool {tool_name}, using empty schema")
                        except Exception as e:
                            if run_logger:
                                run_logger.warning(f"Could not extract schema for tool {tool_name}: {e}")
                            # Provide minimal parameters structure as fallback
                            tool_spec["function"]["parameters"] = {
                                "type": "object",
                                "properties": {},
                                "required": []
                            }
                    else:
                        # Always provide parameters structure, even if empty
                        tool_spec["function"]["parameters"] = {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    
                    # Validate the tool spec before adding
                    if ("type" in tool_spec and 
                        "function" in tool_spec and 
                        "name" in tool_spec["function"] and 
                        "description" in tool_spec["function"] and
                        "parameters" in tool_spec["function"]):
                        openai_tools.append(tool_spec)
                        if run_logger:
                            run_logger.info(f"Successfully converted tool: {tool_name}")
                    else:
                        if run_logger:
                            run_logger.error(f"Tool spec validation failed for {tool_name}: {tool_spec}")
                
                if openai_tools:
                    call_params["tools"] = openai_tools
                    if run_logger:
                        run_logger.info(f"Added {len(openai_tools)} tools to LiteLLM call")
                        # Debug: log the actual tool specs being sent
                        for i, tool_spec in enumerate(openai_tools[:2]):  # Log first 2 tools
                            run_logger.info(f"Tool {i}: {json.dumps(tool_spec, indent=2)}")
                else:
                    if run_logger:
                        run_logger.warning(f"No valid tools to add to LiteLLM call (started with {len(tools)} tools)")
            
            response = await acompletion(**call_params)

            if run_logger:
                run_logger.info(f"LiteLLM call completed successfully")
                run_logger.info(f"Response type: {type(response)}")
                run_logger.info(f"Response: {response}")

            # Handle tool calls and extract content
            content = ""
            tools_used = []
            
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                message = getattr(choice, 'message', None)
                
                if message:
                    # Extract text content
                    content = getattr(message, 'content', "") or ""
                    
                    # Handle tool calls if present - use the O3 pattern
                    tool_calls = getattr(message, 'tool_calls', None)
                    if tool_calls and mcp_client:
                        if run_logger:
                            run_logger.info(f"Model requested {len(tool_calls)} tool calls")
                        
                        # Execute all tool calls and collect results
                        tool_results = []
                        for tool_call in tool_calls:
                            tool_name = tool_call.function.name
                            tool_args = {}
                            
                            try:
                                tool_args = json.loads(tool_call.function.arguments)
                            except Exception as e:
                                # Attempt to salvage JSON args from assistant content
                                recovered_args = {}
                                try:
                                    import re
                                    content_text = getattr(message, 'content', '') or ''
                                    # Find the last JSON-like object in the text
                                    candidates = re.findall(r"\{[\s\S]*?\}", content_text)
                                    if candidates:
                                        for candidate in reversed(candidates):
                                            try:
                                                recovered_args = json.loads(candidate)
                                                break
                                            except Exception:
                                                continue
                                except Exception:
                                    recovered_args = {}
                                if recovered_args:
                                    tool_args = recovered_args
                                    if run_logger:
                                        run_logger.info(f"Recovered tool args from assistant content for {tool_name}: {str(tool_args)[:200]}...")
                                else:
                                    if run_logger:
                                        run_logger.warning(f"Could not parse tool arguments for {tool_name}: {e}")
                                    continue
                            
                            if run_logger:
                                run_logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                            
                            try:
                                # Find the matching MCP tool
                                mcp_tool = None
                                for tool in tools:
                                    if hasattr(tool, 'name') and (
                                        tool.name == tool_name or tool.name.endswith(f"_{tool_name}")
                                    ):
                                        mcp_tool = tool
                                        break
                                
                                if mcp_tool:
                                    # Execute the MCP tool
                                    tool_result = await mcp_tool.ainvoke(tool_args)
                                    tools_used.append(tool_name)
                                    
                                    if run_logger:
                                        run_logger.info(f"Tool {tool_name} executed successfully: {str(tool_result)[:200]}...")
                                    
                                    # Add to tool results for follow-up call
                                    tool_results.append({
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "content": str(tool_result)
                                    })
                                else:
                                    if run_logger:
                                        run_logger.warning(f"Could not find MCP tool: {tool_name}")
                                    tool_results.append({
                                        "tool_call_id": tool_call.id,
                                        "role": "tool", 
                                        "content": f"Error: Tool {tool_name} not found"
                                    })
                                    
                            except Exception as e:
                                if run_logger:
                                    run_logger.error(f"Tool execution failed for {tool_name}: {e}")
                                tool_results.append({
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "content": f"Error: {str(e)}"
                                })
                        
                        # Make follow-up call with tool results (O3 pattern)
                        if tool_results:
                            if run_logger:
                                run_logger.info(f"Making follow-up call with {len(tool_results)} tool results")
                            
                            follow_up_messages = [
                                messages[0],  # system message
                                messages[1],  # user message  
                                {
                                    "role": "assistant",
                                    "content": message.content,
                                    "tool_calls": [{
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments
                                        }
                                    } for tc in tool_calls]
                                },
                                *tool_results
                            ]
                            
                            follow_up_response = await acompletion(
                                model=litellm_model,
                                messages=follow_up_messages,
                                api_key=OPENROUTER_KEY,
                                api_base="https://openrouter.ai/api/v1",
                                temperature=0.7,
                                stream=False
                            )
                            
                            if follow_up_response.choices:
                                content = follow_up_response.choices[0].message.content or ""
                                if run_logger:
                                    run_logger.info(f"Follow-up response: {content[:500]}...")
                    else:
                        # Backward compatibility: single function_call
                        single_function_call = getattr(message, 'function_call', None)
                        if single_function_call and mcp_client:
                            if run_logger:
                                run_logger.info("Model returned legacy single function_call; executing it")
                            try:
                                tool_name = getattr(single_function_call, 'name', None)
                                raw_args = getattr(single_function_call, 'arguments', '{}')
                                try:
                                    tool_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                                except Exception:
                                    tool_args = {}
                                if tool_name:
                                    # Find the matching MCP tool
                                    mcp_tool = None
                                    for tool in tools:
                                        if hasattr(tool, 'name') and tool.name == tool_name:
                                            mcp_tool = tool
                                            break
                                    if mcp_tool:
                                        tool_result = await mcp_tool.ainvoke(tool_args)
                                        tools_used.append(tool_name)
                                        if run_logger:
                                            run_logger.info(f"Legacy function_call tool {tool_name} executed successfully")
                                        # Follow-up call to let the model finalize
                                        follow_up_messages = [
                                            messages[0],
                                            messages[1],
                                            {
                                                "role": "assistant",
                                                "content": message.content,
                                                "tool_calls": [{
                                                    "id": "call_0",
                                                    "type": "function",
                                                    "function": {"name": tool_name, "arguments": json.dumps(tool_args)}
                                                }]
                                            },
                                            {
                                                "tool_call_id": "call_0",
                                                "role": "tool",
                                                "content": str(tool_result)
                                            }
                                        ]
                                        follow_up_response = await acompletion(
                                            model=litellm_model,
                                            messages=follow_up_messages,
                                            api_key=OPENROUTER_KEY,
                                            api_base="https://openrouter.ai/api/v1",
                                            temperature=0.7,
                                            stream=False
                                        )
                                        if follow_up_response.choices:
                                            content = follow_up_response.choices[0].message.content or ""
                                    else:
                                        if run_logger:
                                            run_logger.warning(f"Legacy function_call tool not found: {tool_name}")
                            except Exception as e:
                                if run_logger:
                                    run_logger.error(f"Legacy function_call execution failed: {e}")
                    
                elif hasattr(choice, 'text'):
                    content = choice.text or ""

            if run_logger:
                run_logger.info(f"Extracted content length: {len(content)}")
                run_logger.info(f"Tools executed: {tools_used}")
                if content:
                    run_logger.info(f"Content preview: {content[:500]}{'...' if len(content) > 500 else ''}")

            # Extract token usage from LiteLLM response
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0

            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                if hasattr(usage, 'prompt_tokens'):
                    input_tokens = usage.prompt_tokens
                if hasattr(usage, 'completion_tokens'):
                    output_tokens = usage.completion_tokens
                if hasattr(usage, 'total_tokens'):
                    total_tokens = usage.total_tokens
                
                if run_logger:
                    run_logger.info(f"Token usage from LiteLLM: input={input_tokens}, output={output_tokens}, total={total_tokens}")
            
            # Fallback token counting if not provided
            if total_tokens == 0:
                if content:
                    output_tokens = self.count_tokens_directly(content, model, run_logger)
                    input_text = messages[0]["content"] + " " + messages[1]["content"]
                    input_tokens = self.count_tokens_directly(input_text, model, run_logger)
                    total_tokens = input_tokens + output_tokens
                    
                    if run_logger:
                        run_logger.info(f"Fallback token counting: input={input_tokens}, output={output_tokens}, total={total_tokens}")

            # Calculate metrics
            latency = time.time() - start_time

            # tools_used is already populated from actual tool execution above
            # Only fall back to text detection if no tools were actually executed
            if not tools_used:
                tools_used = self.detect_tool_usage_in_text(content or "", tools)

            # Calculate accuracy score
            accuracy = self.calculate_accuracy_score(query, content or "")

            # Calculate projected cost using LiteLLM
            projected_cost = 0.0
            cost_error = None
            try:
                if hasattr(response, 'usage') and response.usage:
                    # Try to calculate cost using LiteLLM's built-in cost calculator
                    projected_cost = litellm.completion_cost(
                        completion_response=response,
                        model=litellm_model
                    )
                    if run_logger:
                        run_logger.info(f"LiteLLM calculated cost: ${projected_cost:.6f}")
                elif input_tokens > 0 or output_tokens > 0:
                    # Fallback: calculate cost manually using token counts
                    projected_cost = litellm.completion_cost(
                        model=litellm_model,
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens
                    )
                    if run_logger:
                        run_logger.info(f"LiteLLM calculated cost (manual): ${projected_cost:.6f}")
            except Exception as e:
                cost_error = str(e)
                if run_logger:
                    run_logger.warning(f"Could not calculate cost: {cost_error}")

            final_result = {
                "success": True,
                "response": content,
                "error": None,
                "metrics": {
                    "latency": round(latency, 3),
                    "accuracy": accuracy,
                    "tools_used_count": len(tools_used),
                    "tools_used_list": tools_used,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "projected_cost": round(projected_cost, 8) if projected_cost > 0 else 0.0,
                    "cost_error": cost_error,
                },
            }
            
            if run_logger:
                run_logger.info(f"Query completed successfully")
                run_logger.info(f"Final metrics: {final_result['metrics']}")
                run_logger.info(f"Tools used: {tools_used}")
                run_logger.info(f"Accuracy score: {accuracy}")
            
            return final_result

        except Exception as e:
            latency = time.time() - start_time
            error_result = {
                "success": False,
                "response": None,
                "error": str(e),
                "metrics": {
                    "latency": round(latency, 3),
                    "accuracy": 0.0,
                    "tools_used_count": 0,
                    "tools_used_list": [],
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "projected_cost": 0.0,
                    "cost_error": f"Request failed: {str(e)}",
                },
            }
            
            if run_logger:
                run_logger.error(f"Query failed with error: {str(e)}")
                run_logger.error(f"Error type: {type(e)}")
                run_logger.exception("Full exception details:")
            
            return error_result

    def _model_supports_functions(self, model: str) -> bool:
        """Check if model supports function calling"""
        function_capable_models = [
            # OpenAI / OpenRouter (explicit)
            "openai/gpt-5",
            "openrouter/openai/gpt-5",
            "openai/o3",
            "openrouter/openai/o3",
            "openai/o3-mini",
            "openai/gpt-4",
            "openai/gpt-4-turbo",
            # Anthropic
            "anthropic/claude-sonnet-4",
            # Google
            "google/gemini",
        ]
        return any(capable_model in model for capable_model in function_capable_models)

    async def _prefix_tools_with_server_names(
        self,
        raw_tools: List[Any],
        mcp_client: MultiServerMCPClient,
        server_configs: Dict[str, Any] = None,
    ) -> List[Any]:
        """Prefix tool names with their server names to ensure uniqueness"""
        try:
            from copy import deepcopy

            if server_configs:
                console.print(
                    f"[cyan]Available server names for prefixing: {list(server_configs.keys())}[/cyan]"
                )

            prefixed_tools = []
            tool_name_counts = {}

            # First pass: count occurrences of each tool name
            for tool in raw_tools:
                tool_name = getattr(tool, "name", str(tool))
                tool_name_counts[tool_name] = tool_name_counts.get(tool_name, 0) + 1

            # Second pass: prefix duplicates with server names
            server_tool_counters = {}

            for tool in raw_tools:
                tool_name = getattr(tool, "name", str(tool))

                # If this tool name appears only once, keep it as-is
                if tool_name_counts[tool_name] == 1:
                    prefixed_tools.append(tool)
                else:
                    # This is a duplicate - we need to prefix it
                    server_name = self._get_tool_server_name(
                        tool, mcp_client, server_configs
                    )

                    if not server_name:
                        # Fallback: use a counter for this tool name
                        counter_key = tool_name
                        server_tool_counters[counter_key] = (
                            server_tool_counters.get(counter_key, 0) + 1
                        )
                        server_name = f"server{server_tool_counters[counter_key]}"

                    # Create a copy of the tool with prefixed name
                    prefixed_tool = deepcopy(tool)
                    original_name = tool_name
                    prefixed_name = f"{server_name}_{tool_name}"

                    # Update the tool name
                    if hasattr(prefixed_tool, "name"):
                        prefixed_tool.name = prefixed_name

                    # Also update the tool's internal name in its schema if present
                    if hasattr(prefixed_tool, "args_schema") and hasattr(
                        prefixed_tool.args_schema, "__name__"
                    ):
                        prefixed_tool.args_schema.__name__ = prefixed_name

                    prefixed_tools.append(prefixed_tool)
                    console.print(
                        f"[cyan]Prefixed duplicate tool: {original_name} -> {prefixed_name}[/cyan]"
                    )

            return prefixed_tools

        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not prefix tools ({str(e)}), returning original tools[/yellow]"
            )
            return raw_tools

    def _get_tool_server_name(
        self,
        tool: Any,
        mcp_client: MultiServerMCPClient,
        server_configs: Dict[str, Any] = None,
    ) -> Optional[str]:
        """Try to determine which server a tool belongs to"""
        try:
            # Check if tool has server information in its attributes
            if hasattr(tool, "server_name"):
                return tool.server_name

            if hasattr(tool, "_server_name"):
                return tool._server_name

            if hasattr(tool, "metadata") and isinstance(tool.metadata, dict):
                if "server" in tool.metadata:
                    return tool.metadata["server"]
                if "server_name" in tool.metadata:
                    return tool.metadata["server_name"]

            # If we have server configs, use the actual server names from the configuration
            if server_configs:
                server_names = list(server_configs.keys())

                # Check if the tool description contains any of the actual server names
                if hasattr(tool, "description") and tool.description:
                    description_lower = tool.description.lower()
                    for server_name in server_names:
                        if server_name.lower() in description_lower:
                            return server_name

                # Check if the tool name contains any of the server names
                tool_name = getattr(tool, "name", str(tool)).lower()
                for server_name in server_names:
                    if server_name.lower() in tool_name:
                        return server_name

            return None

        except Exception:
            return None

    def count_tokens_directly(self, text: str, model: str, run_logger: Optional[logging.Logger] = None) -> int:
        """Count tokens directly using tokenizer libraries"""
        if not text:
            return 0
        
        try:
            # For OpenAI/GPT models, use tiktoken
            if TIKTOKEN_AVAILABLE and any(model_prefix in model.lower() for model_prefix in ['gpt', 'openai', 'o1', 'OpenAI o3']):
                try:
                    # Map model names to tiktoken encodings
                    if 'gpt-4' in model.lower():
                        encoding = tiktoken.encoding_for_model("gpt-4")
                    elif 'gpt-3.5' in model.lower():
                        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
                    elif any(prefix in model.lower() for prefix in ['o1', 'OpenAI o3']):
                        # Use gpt-4 encoding for newer models
                        encoding = tiktoken.encoding_for_model("gpt-4")
                    else:
                        # Default to cl100k_base for most OpenAI models
                        encoding = tiktoken.get_encoding("cl100k_base")
                    
                    token_count = len(encoding.encode(text))
                    if run_logger:
                        run_logger.info(f"Counted {token_count} tokens using tiktoken for model {model}")
                    return token_count
                except Exception as e:
                    if run_logger:
                        run_logger.warning(f"tiktoken failed for {model}: {e}, falling back to approximation")
            
            # For other models, use tiktoken cl100k_base as approximation or fallback to character-based
            if TIKTOKEN_AVAILABLE:
                try:
                    encoding = tiktoken.get_encoding("cl100k_base")
                    token_count = len(encoding.encode(text))
                    if run_logger:
                        run_logger.info(f"Estimated {token_count} tokens for {model} using cl100k_base approximation")
                    return token_count
                except Exception:
                    pass
            
            # Ultimate fallback: rough character-based approximation
            token_count = max(1, len(text) // 4)
            if run_logger:
                run_logger.info(f"Fallback: estimated {token_count} tokens for {model} using character approximation")
            return token_count
            
        except Exception as e:
            if run_logger:
                run_logger.error(f"Token counting failed for {model}: {e}")
            return max(1, len(text) // 4)  # Fallback approximation

    def detect_tool_usage_in_text(self, response: str, tools: List[Any]) -> List[str]:
        """Detect which tools were mentioned in the response text"""
        tools_used = []
        response_lower = response.lower()

        for tool in tools:
            tool_name = getattr(tool, "name", str(tool))
            if tool_name:
                tool_name_lower = tool_name.lower()
                # Check for exact tool name match
                if tool_name_lower in response_lower:
                    tools_used.append(tool_name)
                # Also check for the original unprefixed name (in case response mentions it)
                elif (
                    "_" in tool_name
                    and tool_name.split("_", 1)[1].lower() in response_lower
                ):
                    # If the tool was prefixed, also check for the original name
                    original_name = tool_name.split("_", 1)[1]
                    if original_name.lower() in response_lower:
                        tools_used.append(tool_name)  # Still report the prefixed name

        return tools_used

    def calculate_accuracy_score(self, query: str, response: str) -> float:
        """Calculate a simple accuracy score based on response quality"""
        if not response:
            return 0.0

        # Simple heuristics for accuracy scoring
        score = 0.0

        # Length-based scoring (responses should be substantial but not excessive)
        response_length = len(response.split())
        if 50 <= response_length <= 500:
            score += 0.3
        elif 20 <= response_length < 50:
            score += 0.2
        elif response_length > 500:
            score += 0.1

        # Check for structured response
        if any(marker in response for marker in ["1.", "2.", "-", "*", "##", "**"]):
            score += 0.2

        # Check for specific content quality indicators
        quality_indicators = [
            "because",
            "therefore",
            "however",
            "specifically",
            "according to",
            "based on",
            "analysis",
            "result",
            "conclusion",
            "evidence",
        ]
        matching_indicators = sum(
            1 for indicator in quality_indicators if indicator in response.lower()
        )
        score += min(matching_indicators * 0.1, 0.3)

        # Check if response appears to address the query
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words)) / max(
            len(query_words), 1
        )
        score += min(overlap, 0.2)

        return min(score, 1.0)

    async def process_single_model(
        self,
        model: str,
        query_text: str,
        expected_outcome: str,
        battery_name: str,
        cached_tools: List[Any],
        mcp_client: Optional[MultiServerMCPClient] = None,
    ) -> Dict[str, Any]:
        """Process a single model with all retry attempts concurrently controlled by semaphore"""
        async with self.semaphore:  # Control concurrent requests
            console.print(f"[yellow]Testing model: {model}[/yellow]")

            # Run multiple attempts with retry logic
            runs = []
            for attempt in range(retry_count):
                timestamp = datetime.now().isoformat()
                query_id = self.generate_query_id(query_text, model, timestamp)
                run_hash = self.generate_run_hash(
                    query_text, model, battery_name, timestamp
                )

                # Setup run-specific logging
                run_logger, log_file_path = self.setup_run_logging(query_id)
                run_logger.info(
                    f"Starting run {attempt + 1}/{retry_count} for model {model}"
                )
                run_logger.info(f"Query: {query_text}")
                run_logger.info(f"Expected outcome: {expected_outcome}")
                run_logger.info(f"Log file: {log_file_path}")
                run_logger.info(f"Available tools count: {len(cached_tools)}")
                
                # Log tool details
                if cached_tools:
                    tool_names = [getattr(tool, 'name', str(tool)) for tool in cached_tools]
                    run_logger.info(f"Available tools: {tool_names}")
                
                # Query the model with cached tools and logger
                result = await self.query_model(
                    query_text, model, cached_tools, mcp_client, run_logger
                )

                # Create run result
                run_result = RunResult(
                    query=query_text,
                    model=model,
                    experiment=battery_name,
                    timestamp=timestamp,
                    query_id=query_id,
                    run_hash=run_hash,
                    success=result["success"],
                    response=result["response"],
                    error=result["error"],
                    metrics=result["metrics"],
                )

                runs.append(asdict(run_result))

                # Log detailed run result
                if result["success"]:
                    run_logger.info(f"="*50)
                    run_logger.info(f"RUN COMPLETED SUCCESSFULLY")
                    run_logger.info(f"="*50)
                    run_logger.info(
                        f"Response length: {len(result['response'] or '')} characters"
                    )
                    if result['response']:
                        run_logger.info(f"Full response: {result['response']}")
                    run_logger.info(f"Complete metrics: {json.dumps(result['metrics'], indent=2)}")
                    run_logger.info(f"Total tokens used: {result['metrics'].get('total_tokens', 0)}")
                    run_logger.info(f"Input tokens: {result['metrics'].get('input_tokens', 0)}")
                    run_logger.info(f"Output tokens: {result['metrics'].get('output_tokens', 0)}")
                    run_logger.info(f"Tools used: {result['metrics'].get('tools_used_list', [])}")
                    console.print(
                        f"  [green]✓ Attempt {attempt + 1}: Success (latency: {result['metrics']['latency']}s, tokens: {result['metrics'].get('total_tokens', 0)})[/green]"
                    )
                else:
                    run_logger.error(f"="*50)
                    run_logger.error(f"RUN FAILED")
                    run_logger.error(f"="*50)
                    run_logger.error(f"Error: {result['error']}")
                    run_logger.error(f"Metrics at failure: {json.dumps(result['metrics'], indent=2)}")
                    console.print(
                        f"  [red]✗ Attempt {attempt + 1}: Failed - {result['error'][:100]}[/red]"
                    )
                
                # Ensure all log data is flushed
                for handler in run_logger.handlers:
                    handler.flush()
                
                # Clean up this run's logger handlers to prevent file descriptor leaks
                litellm_logger = logging.getLogger("litellm")
                for handler in run_logger.handlers:
                    if handler in litellm_logger.handlers:
                        litellm_logger.removeHandler(handler)
                    
                # Properly close file handlers to prevent file descriptor leaks
                for handler in run_logger.handlers[:]:
                    handler.close()
                    run_logger.removeHandler(handler)

                # Small delay between attempts
                if attempt < retry_count - 1:
                    await asyncio.sleep(1)

            # Calculate model summary
            successful_runs = [r for r in runs if r["success"]]
            total_runs = len(runs)
            success_rate = (
                len(successful_runs) / total_runs if total_runs > 0 else 0
            )

            if successful_runs:
                avg_latency = sum(
                    r["metrics"]["latency"] for r in successful_runs
                ) / len(successful_runs)
                avg_accuracy = sum(
                    r["metrics"]["accuracy"] for r in successful_runs
                ) / len(successful_runs)
                average_tools_used = sum(
                    r["metrics"]["tools_used_count"] for r in successful_runs
                ) / len(successful_runs)
                average_input_tokens = sum(
                    r["metrics"]["input_tokens"] for r in successful_runs
                ) / len(successful_runs)
                average_output_tokens = sum(
                    r["metrics"]["output_tokens"] for r in successful_runs
                ) / len(successful_runs)
                average_total_tokens = sum(
                    r["metrics"].get("total_tokens", r["metrics"].get("input_tokens", 0) + r["metrics"].get("output_tokens", 0)) for r in successful_runs
                ) / len(successful_runs)
                avg_cost = sum(
                    r["metrics"].get("projected_cost", 0) for r in successful_runs
                ) / len(successful_runs)
            else:
                avg_latency = avg_accuracy = average_tools_used = average_input_tokens = average_output_tokens = average_total_tokens = avg_cost = 0

            model_summary = ModelSummary(
                success_rate=success_rate,
                successful_runs=len(successful_runs),
                total_runs=total_runs,
                avg_latency=round(avg_latency, 3),
                avg_accuracy=round(avg_accuracy, 2),
                tools_used=round(average_tools_used, 0),
                input_tokens=round(average_input_tokens, 0),
                output_tokens=round(average_output_tokens, 0),
                total_tokens=round(average_total_tokens, 0),
                avg_cost=round(avg_cost, 8),
            )

            console.print(
                f"  [blue]Summary: {success_rate:.1%} success rate, avg latency: {avg_latency:.2f}s, avg tokens: {average_input_tokens:.0f}in/{average_output_tokens:.0f}out/{average_total_tokens:.0f}total, avg cost: ${avg_cost:.6f}[/blue]"
            )

            return {"model": model, "runs": runs, "summary": asdict(model_summary)}

    async def run_battery(
        self, battery: Dict[str, Any], specific_test_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """Run a complete test battery"""
        battery_name = battery["name"]
        models = battery["candidate_models"]
        mcp_server_configs = battery.get("mcp_servers", {})
        queries = battery["queries"]

        console.print(f"\n[bold blue]Running battery: {battery_name}[/bold blue]")
        console.print(f"Models: {models}")
        console.print(f"MCP servers: {len(mcp_server_configs)}")
        console.print(f"Queries: {len(queries)}")

        # Setup MCP client using LangChain
        mcp_client = await self.setup_mcp_client(mcp_server_configs)

        # Cache tools once per battery to avoid restarting MCP servers
        cached_tools = []
        if mcp_client:
            try:
                console.print(
                    f"[blue]Fetching MCP tools once for entire battery...[/blue]"
                )
                raw_tools = await asyncio.wait_for(mcp_client.get_tools(), timeout=30.0)

                # Prefix tools with server names to ensure uniqueness
                cached_tools = await self._prefix_tools_with_server_names(
                    raw_tools, mcp_client, mcp_server_configs
                )

                console.print(
                    f"[green]Cached {len(raw_tools)} tools, created {len(cached_tools)} uniquely named tools for battery[/green]"
                )
            except asyncio.TimeoutError:
                console.print(
                    f"[yellow]Warning: MCP tool retrieval timed out - proceeding without tools[/yellow]"
                )
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Could not retrieve MCP tools ({str(e)[:100]}) - proceeding without tools[/yellow]"
                )
                import traceback
                console.print(f"[red]Detailed error: {traceback.format_exc()}[/red]")
        else:
            console.print(
                f"[yellow]Warning: No MCP client available - proceeding without tools[/yellow]"
            )

        try:
            # Prepare queries
            query_list = []
            if isinstance(queries, dict):
                # Handle dictionary format (query -> expected_outcome)
                for i, (query_text, expected_outcome) in enumerate(queries.items()):
                    if specific_test_index is None or i == specific_test_index:
                        query_list.append(
                            {
                                "query": query_text,
                                "expected_outcome": expected_outcome,
                                "query_index": i,
                            }
                        )
            else:
                # Handle list format
                for i, query_item in enumerate(queries):
                    if specific_test_index is None or i == specific_test_index:
                        if isinstance(query_item, str):
                            query_list.append(
                                {
                                    "query": query_item,
                                    "expected_outcome": "Response should be relevant and accurate",
                                    "query_index": i,
                                }
                            )
                        else:
                            query_list.append({"query_index": i, **query_item})

            # Process each query
            processed_queries = []

            for query_data in query_list:
                query_text = query_data["query"]
                query_index = query_data["query_index"]
                expected_outcome = query_data.get(
                    "expected_outcome", "Response should be relevant and accurate"
                )

                console.print(
                    f"\n[cyan]Query {query_index}: {query_text[:100]}{'...' if len(query_text) > 100 else ''}[/cyan]"
                )

                # Process all models concurrently using asyncio.gather
                console.print(f"[blue]Processing {len(models)} models concurrently (max {self.semaphore._value} concurrent requests)...[/blue]")
                
                concurrent_start_time = time.time()
                model_tasks = [
                    self.process_single_model(
                        model, query_text, expected_outcome, battery_name, cached_tools, mcp_client
                    )
                    for model in models
                ]
                
                # Execute all model tasks concurrently
                model_results = await asyncio.gather(*model_tasks, return_exceptions=True)
                concurrent_duration = time.time() - concurrent_start_time
                console.print(f"[green]Concurrent processing completed in {concurrent_duration:.2f}s[/green]")
                
                # Handle any exceptions that occurred during concurrent processing
                final_model_results = []
                for i, result in enumerate(model_results):
                    if isinstance(result, Exception):
                        console.print(f"[red]Model {models[i]} failed with exception: {result}[/red]")
                        # Create a failed result for this model
                        failed_result = {
                            "model": models[i],
                            "runs": [],
                            "summary": asdict(ModelSummary(
                                success_rate=0.0,
                                successful_runs=0,
                                total_runs=0,
                                avg_latency=0.0,
                                avg_accuracy=0.0,
                                tools_used=0.0,
                                input_tokens=0.0,
                                output_tokens=0.0,
                                total_tokens=0.0,
                                avg_cost=0.0,
                            ))
                        }
                        final_model_results.append(failed_result)
                    else:
                        final_model_results.append(result)
                
                model_results = final_model_results

                processed_queries.append(
                    {
                        "query": query_text,
                        "expected_outcome": expected_outcome,
                        "query_index": query_index,
                        "models": model_results,
                    }
                )

            # Create server info for compatibility
            mcp_server_info = {}
            if mcp_client:
                try:
                    # Use the cached tools to determine connected servers
                    for server_name in mcp_server_configs.keys():
                        # Filter tools that belong to this specific server
                        server_tools = []
                        for tool in cached_tools:
                            if hasattr(tool, "name"):
                                tool_name = tool.name
                                # Check if tool belongs to this server (either prefixed or determined by server name)
                                if tool_name.startswith(f"{server_name}_"):
                                    server_tools.append(tool_name)
                                else:
                                    # Check if this tool belongs to the server based on server name detection
                                    detected_server = self._get_tool_server_name(tool, mcp_client, mcp_server_configs)
                                    if detected_server == server_name:
                                        server_tools.append(tool_name)
                        
                        mcp_server_info[server_name] = MCPServerInfo(
                            name=server_name,
                            connected=len(server_tools) > 0,
                            tools=server_tools,
                            resources=0,
                            prompts=0,
                        ).to_dict()
                except Exception:
                    # Fallback for server info when tool retrieval fails
                    for server_name in mcp_server_configs.keys():
                        mcp_server_info[server_name] = MCPServerInfo(
                            name=server_name, connected=False
                        ).to_dict()
            else:
                # No MCP client available
                for server_name in mcp_server_configs.keys():
                    mcp_server_info[server_name] = MCPServerInfo(
                        name=server_name, connected=False
                    ).to_dict()

            return {
                "name": battery_name,
                "models": models,
                "mcp_servers": mcp_server_info,
                "queries": processed_queries,
            }

        finally:
            # Clean up MCP client
            await self.cleanup_mcp_client(mcp_client)

    async def run_experiments(
        self,
        battery_name: Optional[str] = None,
        specific_test_index: Optional[int] = None,
    ):
        """Run all experiments or a specific battery"""
        start_time = datetime.now()
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")

        console.print(f"[bold green]Starting Osmosis MCP Experiment Suite (LiteLLM Version)[/bold green]")
        console.print(f"Timestamp: {timestamp}")
        console.print(f"Output directory: {self.output_dir}")

        # Filter batteries
        batteries_to_run = test_batteries
        if battery_name:
            batteries_to_run = [b for b in test_batteries if b["name"] == battery_name]
            if not batteries_to_run:
                console.print(f"[red]Battery '{battery_name}' not found![/red]")
                return

        results = {
            "batteries": [],
            "metadata": {
                "timestamp": timestamp,
                "experiment_start": start_time.isoformat(),
                "retry_count": retry_count,
                "version": "litellm",
            },
        }

        # Run each battery
        for battery in batteries_to_run:
            try:
                battery_result = await self.run_battery(battery, specific_test_index)
                results["batteries"].append(battery_result)
            except Exception as e:
                console.print(
                    f"[red]Failed to run battery {battery['name']}: {e}[/red]"
                )
                self.logger.exception(f"Battery {battery['name']} failed")

        # Add end timestamp
        end_time = datetime.now()
        results["metadata"]["experiment_end"] = end_time.isoformat()

        # Save results with explicit file handle management
        output_file = self.output_dir / f"{timestamp}_litellm.json"
        try:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except OSError as e:
            if e.errno == 24:  # Too many open files
                console.print(f"[red]ERROR: Too many open files when saving results. File descriptor leak detected![/red]")
                console.print(f"[red]Try reducing --max-concurrent or fix resource cleanup issues.[/red]")
                # Force garbage collection and try again
                import gc
                gc.collect()
                await asyncio.sleep(1)
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                raise

        console.print(f"\n[bold green]Experiment completed![/bold green]")
        console.print(f"Results saved to: {output_file}")
        console.print(f"Total duration: {end_time - start_time}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Osmosis MCP Experiment Runner (LiteLLM Version) - Now with Concurrent Processing!")
    parser.add_argument("--battery", help="Run specific battery by name")
    parser.add_argument(
        "--test", type=int, help="Run specific test index within battery"
    )
    parser.add_argument(
        "--output-dir", help="Specify custom output directory (default: output)"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=5, 
        help="Maximum number of concurrent requests (default: 5)"
    )
    args = parser.parse_args()

    runner = ExperimentRunner(output_dir=args.output_dir, max_concurrent_requests=args.max_concurrent)
    asyncio.run(runner.run_experiments(args.battery, args.test))


if __name__ == "__main__":
    main()