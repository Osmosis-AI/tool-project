#!/usr/bin/env python3
import json
import asyncio
from datetime import datetime
from typing import Dict, Set
from termcolor import colored
import colorama
from experiment.experiment_control import test_batteries
from env import SMITHERY_KEY, SMITHERY_PROFILE

# MCP imports
import mcp
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic.networks import AnyUrl

# Initialize colorama for Windows compatibility
colorama.init()

def make_json_serializable(obj):
    """Convert objects to JSON-serializable format"""
    if hasattr(obj, 'model_dump'):
        return make_json_serializable(obj.model_dump())
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, AnyUrl):
        return str(obj)
    elif hasattr(obj, '__dict__'):
        return make_json_serializable(obj.__dict__)
    else:
        # For any other object types that might not be serializable
        try:
            json.dumps(obj)  # Test if it's serializable
            return obj
        except (TypeError, ValueError):
            return str(obj)  # Convert to string as fallback

class MCPServer:
    """Wrapper class to manage individual MCP server instances using proper MCP client"""
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.url = self.config.get("url")
        self.tools = []
        self.resources = []
        self.prompts = []
        self._started = False
        self.error = None
    
    async def start(self) -> bool:
        """Start the MCP server and initialize connection"""
        if self._started:
            return True
            
        try:
            print(colored(f"🔌 Connecting to MCP server: {self.name}", "blue"))
            
            # Check if server is disabled
            if self.config.get("disabled", False):
                print(colored(f"  ⏭️ Skipping {self.name}: server disabled", "yellow"))
                return False
            
            # Check if URL is provided
            if not self.url:
                print(colored(f"  ⏭️ Skipping {self.name}: no URL provided", "yellow"))
                return False
            
            # Perform a health check first
            if not await self._health_check():
                print(colored(f"  ❌ Skipping {self.name}: server health check failed", "red"))
                return False
            
            async with streamablehttp_client(self.url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    
                    try:
                        tools_result = await asyncio.wait_for(session.list_tools(), timeout=10.0)
                        self.tools = [make_json_serializable(tool.model_dump()) for tool in tools_result.tools]
                        print(colored(f"  ✅ Found {len(self.tools)} tools", "green"))
                    except asyncio.TimeoutError:
                        print(colored(f"  ⚠️ Timeout listing tools", "yellow"))
                        self.error = "Timeout listing tools"
                    except Exception as e:
                        print(colored(f"  ⚠️ No tools available: {e}", "yellow"))
                        self.error = f"No tools: {e}"

                    try:
                        resources_result = await asyncio.wait_for(session.list_resources(), timeout=10.0)
                        self.resources = [make_json_serializable(resource.model_dump()) for resource in resources_result.resources]
                        print(colored(f"  ✅ Found {len(self.resources)} resources", "green"))
                    except asyncio.TimeoutError:
                        print(colored(f"  ⚠️ Timeout listing resources", "yellow"))
                    except Exception as e:
                        print(colored(f"  ⚠️ No resources available: {e}", "yellow"))
                        
                    try:
                        prompts_result = await asyncio.wait_for(session.list_prompts(), timeout=10.0)
                        self.prompts = [make_json_serializable(prompt.model_dump()) for prompt in prompts_result.prompts]
                        print(colored(f"  ✅ Found {len(self.prompts)} prompts", "green"))
                    except asyncio.TimeoutError:
                        print(colored(f"  ⚠️ Timeout listing prompts", "yellow"))
                    except Exception as e:
                        print(colored(f"  ⚠️ No prompts available: {e}", "yellow"))

            self._started = True
            print(colored(f"✅ MCP server {self.name} ready", "green"))
            return True
            
        except Exception as e:
            print(colored(f"❌ Failed to setup MCP server {self.name}: {e}", "red"))
            self.error = str(e)
            self._started = False
            return False
    
    async def _health_check(self) -> bool:
        """Perform a health check on the MCP server"""
        if not self.url:
            print(colored(f"  ❌ Health check failed for {self.name}: no URL configured", "red"))
            return False
            
        try:
            print(colored(f"  🔍 Health check for {self.name}...", "cyan"))
            async with streamablehttp_client(self.url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    # Try to initialize and list tools with a short timeout
                    await asyncio.wait_for(session.initialize(), timeout=15.0)
                    await asyncio.wait_for(session.list_tools(), timeout=10.0)
                    print(colored(f"  ✅ Health check passed for {self.name}", "green"))
                    return True
        except Exception as e:
            print(colored(f"  ❌ Health check failed for {self.name}: {type(e).__name__} - {str(e)}", "red"))
            return False

class MCPInfoCollector:
    """Collects and displays information about all MCP servers from test batteries"""
    
    def __init__(self):
        self.all_servers = {}
        self.server_info = {}
        
    def collect_all_mcp_servers(self) -> Dict[str, Dict]:
        """Collect all unique MCP servers from all test batteries"""
        all_servers = {}
        server_sources = {}  # Track which batteries use which servers
        
        for battery in test_batteries:
            battery_name = battery["name"]
            mcp_servers = battery.get("mcp_servers", {})
            
            for server_name, config in mcp_servers.items():
                if server_name not in all_servers:
                    all_servers[server_name] = config
                    server_sources[server_name] = []
                
                server_sources[server_name].append(battery_name)
        
        print(colored(f"🔍 Found {len(all_servers)} unique MCP servers across {len(test_batteries)} test batteries", "cyan"))
        
        # Print which batteries use which servers
        for server_name, batteries in server_sources.items():
            print(colored(f"  📦 {server_name}: used in {len(batteries)} batteries ({', '.join(batteries)})", "white"))
        
        return all_servers
    
    async def setup_mcp_servers(self, mcp_servers: Dict) -> Dict:
        """Setup and start MCP servers"""
        active_servers = {}
        
        print(colored(f"\n🚀 Connecting to {len(mcp_servers)} MCP servers...", "magenta", attrs=["bold"]))
        
        for server_name, config in mcp_servers.items():
            try:
                server = MCPServer(server_name, config)
                if await server.start():
                    active_servers[server_name] = server
                else:
                    print(colored(f"❌ Failed to start MCP server {server_name}", "red"))
            except Exception as e:
                print(colored(f"❌ Failed to setup MCP server {server_name}: {e}", "red"))
                
        return active_servers
    
    def print_server_summary(self, servers: Dict):
        """Print a summary of all servers"""
        print(colored(f"\n📊 MCP Server Summary", "magenta", attrs=["bold"]))
        print(colored("=" * 60, "magenta"))
        
        total_tools = 0
        total_resources = 0
        total_prompts = 0
        active_servers = 0
        failed_servers = 0
        
        for server_name, server in servers.items():
            if server._started:
                active_servers += 1
                total_tools += len(server.tools)
                total_resources += len(server.resources)
                total_prompts += len(server.prompts)
            else:
                failed_servers += 1
        
        print(colored(f"🔌 Total servers: {len(servers)}", "white"))
        print(colored(f"✅ Active servers: {active_servers}", "green"))
        print(colored(f"❌ Failed servers: {failed_servers}", "red"))
        print(colored(f"🛠️  Total tools: {total_tools}", "cyan"))
        print(colored(f"📂 Total resources: {total_resources}", "cyan"))
        print(colored(f"📝 Total prompts: {total_prompts}", "cyan"))
    
    def print_detailed_server_info(self, servers: Dict):
        """Print detailed information for each server"""
        print(colored(f"\n📋 Detailed MCP Server Information", "magenta", attrs=["bold"]))
        print(colored("=" * 60, "magenta"))
        
        for server_name, server in servers.items():
            print(colored(f"\n🏷️  Server: {server_name}", "yellow", attrs=["bold"]))
            print(colored(f"🔗 URL: {server.url}", "white"))
            
            if not server._started:
                print(colored(f"❌ Status: Failed to connect", "red"))
                if server.error:
                    print(colored(f"⚠️  Error: {server.error}", "red"))
                continue
            
            print(colored(f"✅ Status: Connected", "green"))
            
            # Print tools
            if server.tools:
                print(colored(f"\n🛠️  Tools ({len(server.tools)}):", "cyan", attrs=["bold"]))
                for i, tool in enumerate(server.tools, 1):
                    tool_name = tool.get('name', 'Unknown')
                    tool_desc = tool.get('description', 'No description')
                    print(colored(f"  {i}. {tool_name}", "cyan"))
                    print(colored(f"     {tool_desc}", "white"))
                    
                    # Show input schema if available
                    if 'inputSchema' in tool and tool['inputSchema']:
                        schema = tool['inputSchema']
                        if isinstance(schema, dict) and 'properties' in schema:
                            properties = schema['properties']
                            if properties:
                                print(colored(f"     Parameters:", "yellow"))
                                for param_name, param_def in properties.items():
                                    if isinstance(param_def, dict):
                                        param_type = param_def.get('type', 'unknown')
                                        param_desc = param_def.get('description', 'No description')
                                        print(colored(f"       • {param_name} ({param_type}): {param_desc}", "white"))
                                        
                                        # Show constraints
                                        constraints = []
                                        if 'minimum' in param_def:
                                            constraints.append(f"min: {param_def['minimum']}")
                                        if 'maximum' in param_def:
                                            constraints.append(f"max: {param_def['maximum']}")
                                        if 'enum' in param_def:
                                            constraints.append(f"values: {param_def['enum']}")
                                        
                                        if constraints:
                                            print(colored(f"         Constraints: {', '.join(constraints)}", "yellow"))
            else:
                print(colored(f"\n🛠️  Tools: None", "white"))
            
            # Print resources
            if server.resources:
                print(colored(f"\n📂 Resources ({len(server.resources)}):", "cyan", attrs=["bold"]))
                for i, resource in enumerate(server.resources, 1):
                    resource_uri = resource.get('uri', 'Unknown')
                    resource_name = resource.get('name', 'Unknown')
                    resource_desc = resource.get('description', 'No description')
                    print(colored(f"  {i}. {resource_name} ({resource_uri})", "cyan"))
                    print(colored(f"     {resource_desc}", "white"))
            else:
                print(colored(f"\n📂 Resources: None", "white"))
            
            # Print prompts
            if server.prompts:
                print(colored(f"\n📝 Prompts ({len(server.prompts)}):", "cyan", attrs=["bold"]))
                for i, prompt in enumerate(server.prompts, 1):
                    prompt_name = prompt.get('name', 'Unknown')
                    prompt_desc = prompt.get('description', 'No description')
                    print(colored(f"  {i}. {prompt_name}", "cyan"))
                    print(colored(f"     {prompt_desc}", "white"))
                    
                    # Show prompt arguments if available
                    if 'arguments' in prompt and prompt['arguments']:
                        args = prompt['arguments']
                        if isinstance(args, list) and args:
                            print(colored(f"     Arguments:", "yellow"))
                            for arg in args:
                                if isinstance(arg, dict):
                                    arg_name = arg.get('name', 'unknown')
                                    arg_desc = arg.get('description', 'No description')
                                    arg_required = arg.get('required', False)
                                    req_text = " (required)" if arg_required else " (optional)"
                                    print(colored(f"       • {arg_name}{req_text}: {arg_desc}", "white"))
            else:
                print(colored(f"\n📝 Prompts: None", "white"))
            
            print(colored("-" * 40, "white"))
    
    def save_server_info_to_json(self, servers: Dict):
        """Save server information to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"output/mcp_info_{timestamp}.json"
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs("output", exist_ok=True)
        
        # Prepare data for JSON
        server_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_servers": len(servers),
                "active_servers": sum(1 for s in servers.values() if s._started),
                "failed_servers": sum(1 for s in servers.values() if not s._started),
                "total_tools": sum(len(s.tools) for s in servers.values() if s._started),
                "total_resources": sum(len(s.resources) for s in servers.values() if s._started),
                "total_prompts": sum(len(s.prompts) for s in servers.values() if s._started)
            },
            "servers": {}
        }
        
        for server_name, server in servers.items():
            server_data["servers"][server_name] = {
                "name": server_name,
                "url": server.url,
                "connected": server._started,
                "error": server.error,
                "tools": server.tools if server._started else [],
                "resources": server.resources if server._started else [],
                "prompts": server.prompts if server._started else []
            }
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(make_json_serializable(server_data), f, indent=2, ensure_ascii=False)
        
        print(colored(f"\n💾 Server information saved to: {output_file}", "green", attrs=["bold"]))
    
    async def run(self):
        """Main execution method"""
        print(colored("🔍 MCP Server Information Collector", "magenta", attrs=["bold"]))
        print(colored("=" * 60, "magenta"))
        
        # Check environment variables  
        if not SMITHERY_KEY or not SMITHERY_PROFILE:
            print(colored("⚠️ Warning: SMITHERY_KEY and/or SMITHERY_PROFILE not set", "yellow"))
            print(colored("Some MCP servers may not work without Smithery credentials.", "yellow"))
            print()
        
        # Collect all MCP servers from test batteries
        all_servers_config = self.collect_all_mcp_servers()
        
        # Connect to all servers
        active_servers = await self.setup_mcp_servers(all_servers_config)
        
        # Print summary and detailed information
        self.print_server_summary(active_servers)
        self.print_detailed_server_info(active_servers)
        
        # Save to JSON file
        self.save_server_info_to_json(active_servers)
        
        print(colored(f"\n🎉 MCP server information collection completed!", "green", attrs=["bold"]))

async def main():
    """Main entry point"""
    # Check for environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    collector = MCPInfoCollector()
    await collector.run()

if __name__ == "__main__":
    asyncio.run(main()) 