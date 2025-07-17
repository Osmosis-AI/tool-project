#!/usr/bin/env python3
import json
import os
import time
from datetime import datetime
from typing import Dict
from openai import OpenAI
from termcolor import colored
import colorama
from experiment.experiment_control import test_batteries
from dotenv import load_dotenv


load_dotenv()

# Initialize colorama for Windows compatibility
colorama.init()

class ExperimentRunner:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_KEY"),
        )
        self.results = []
        self.current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def setup_mcp_servers(self, mcp_servers: Dict) -> Dict:
        """Setup and start MCP servers"""
        active_servers = {}
        
        for server_name, config in mcp_servers.items():
            try:
                print(colored(f"Setting up MCP server: {server_name}", "blue"))
                # For this experiment, we'll simulate MCP server integration
                # In a real implementation, you'd start the actual MCP server process
                active_servers[server_name] = {
                    "status": "active",
                    "config": config
                }
                print(colored(f"✓ MCP server {server_name} ready", "green"))
            except Exception as e:
                print(colored(f"✗ Failed to setup MCP server {server_name}: {e}", "red"))
                
        return active_servers
    
    def log_to_console(self, content: str, exp_name: str|None = None, model_name: str|None = None):
        """Log output to console with colored format"""
        if exp_name and model_name:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"[{exp_name}] - [{model_name}] - [{timestamp}] - {content}"
            end_char = "\n"
        else:
            log_line = content
            end_char = ""
        
        # Color coding based on content type
        if "error" in content.lower() or "failed" in content.lower():
            print(colored(log_line, "red"), end=end_char)
        elif "success" in content.lower() or "completed" in content.lower():
            print(colored(log_line, "green"), end=end_char)
        elif "warning" in content.lower():
            print(colored(log_line, "yellow"), end=end_char)
        else:
            print(colored(log_line, "cyan"), end=end_char)
    
    def execute_query(self, query: str, model: str, exp_name: str, mcp_servers: Dict) -> Dict:
        """Execute a single query against a model with MCP server context"""
        
        # Log query start
        self.log_to_console(f"Starting query: {query[:50]}...", exp_name, model)
        
        start_time = time.time()
        result = {
            "query": query,
            "model": model,
            "experiment": exp_name,
            "timestamp": datetime.now().isoformat(),
            "mcp_servers": list(mcp_servers.keys()),
            "success": False,
            "response": None,
            "error": None,
            "duration": 0
        }
        
        try:
            # Enhance the query with MCP server context information
            enhanced_query = query
            if mcp_servers:
                server_context = f"\n\nNote: This query is being processed with the following MCP servers active: {', '.join(mcp_servers.keys())}"
                enhanced_query = query + server_context
            
            # Make the API call with streaming
            completion = self.client.chat.completions.create(
                extra_headers={
                    "X-Title": f"osmosis-experiment-{exp_name}",
                },
                model=model,
                messages=[{
                    "role": "user", 
                    "content": enhanced_query
                }],
                stream=True
            )
            
            # Collect streamed response
            full_response = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    # Log each streamed chunk
                    self.log_to_console(content)
            
            result["response"] = full_response
            result["success"] = True
            result["duration"] = time.time() - start_time
            
            self.log_to_console(f"\n\nQuery completed successfully in {result['duration']:.2f}s", exp_name, model)
            
        except Exception as e:
            result["error"] = str(e)
            result["duration"] = time.time() - start_time
            self.log_to_console(f"Query failed: {str(e)}", exp_name, model)
        
        return result
    
    def run_experiment(self, battery: Dict):
        """Run a complete experiment battery"""
        exp_name = battery["name"]
        models = battery["candidate_models"]
        queries = battery["queries"]
        mcp_servers_config = battery.get("mcp_servers", {})
        
        print(colored(f"\n🚀 Starting experiment: {exp_name}", "magenta", attrs=["bold"]))
        print(colored(f"Models: {', '.join(models)}", "blue"))
        print(colored(f"Queries: {len(queries)}", "blue"))
        print(colored(f"MCP Servers: {', '.join(mcp_servers_config.keys())}", "blue"))
        
        # Setup MCP servers
        active_mcp_servers = self.setup_mcp_servers(mcp_servers_config)
        
        # Run queries for each model
        for model in models:
            print(colored(f"\n📊 Testing model: {model}", "yellow", attrs=["bold"]))
            
            for i, query in enumerate(queries, 1):
                print(colored(f"\n🔍 Query {i}/{len(queries)}", "white", attrs=["bold"]))
                
                result = self.execute_query(query, model, exp_name, active_mcp_servers)
                self.results.append(result)
                
                # Brief pause between queries
                time.sleep(1)
        
        print(colored(f"\n✅ Experiment {exp_name} completed!", "green", attrs=["bold"]))
    
    def save_results(self):
        """Save results to timestamped JSON file"""
        os.makedirs("output", exist_ok=True)
        output_file = f"output/{self.current_timestamp}.json"
        
        experiment_data = {
            "timestamp": self.current_timestamp,
            "experiment_start": datetime.now().isoformat(),
            "total_queries": len(self.results),
            "results": self.results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(experiment_data, f, indent=2, ensure_ascii=False)
        
        print(colored(f"\n💾 Results saved to: {output_file}", "green", attrs=["bold"]))
        
        # Print summary statistics
        successful_queries = sum(1 for r in self.results if r["success"])
        failed_queries = len(self.results) - successful_queries
        avg_duration = sum(r["duration"] for r in self.results) / len(self.results) if self.results else 0
        
        print(colored(f"\n📈 Summary Statistics:", "cyan", attrs=["bold"]))
        print(colored(f"  Total queries: {len(self.results)}", "white"))
        print(colored(f"  Successful: {successful_queries}", "green"))
        print(colored(f"  Failed: {failed_queries}", "red"))
        print(colored(f"  Average duration: {avg_duration:.2f}s", "white"))
    
    def run_all_experiments(self):
        """Run all experiment batteries"""
        print(colored("🧪 Starting MCP Overflow Experiment Suite", "magenta", attrs=["bold"]))
        print(colored("=" * 50, "magenta"))
        
        for battery in test_batteries:
            self.run_experiment(battery)
        
        self.save_results()
        print(colored("\n🎉 All experiments completed!", "green", attrs=["bold"]))


def main():
    """Main entry point for the experiment"""
    if not os.getenv("OPENROUTER_KEY"):
        print(colored("❌ Error: OPENROUTER_KEY environment variable not set", "red", attrs=["bold"]))
        print(colored("Please set your OpenRouter API key:", "yellow"))
        print(colored("export OPENROUTER_KEY='your-api-key-here'", "cyan"))
        return
    
    runner = ExperimentRunner()
    runner.run_all_experiments()

if __name__ == "__main__":
    main()