#!/usr/bin/env python3
"""
Script to parse experiment output JSON and display queries and responses
with colored formatting for easy reading.
"""

import json
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

def extract_tools_from_run(run_data):
    """Extract tools used from run data."""
    tools = []
    
    # Check for MCP servers used (this is where the actual tools are stored)
    if 'mcp_servers' in run_data:
        tools = run_data['mcp_servers']
    # Check various other possible locations for tool information
    elif 'tools' in run_data:
        tools = run_data['tools']
    elif 'tools_used' in run_data:
        tools = run_data['tools_used']
    elif 'metadata' in run_data and 'tools' in run_data['metadata']:
        tools = run_data['metadata']['tools']
    elif 'response_metadata' in run_data and 'tools' in run_data['response_metadata']:
        tools = run_data['response_metadata']['tools']
    
    # Ensure tools is a list
    if isinstance(tools, str):
        tools = [tools]
    elif not isinstance(tools, list):
        tools = []
    
    return tools

def parse_and_display_responses(json_file_path):
    """Parse JSON file and display queries/responses with colors."""
    
    console = Console()
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        console.print(f"[red]Error: File '{json_file_path}' not found.[/red]")
        return
    except json.JSONDecodeError as e:
        console.print(f"[red]Error parsing JSON: {e}[/red]")
        return
    
    # Display file info
    console.print(Rule(f"[bold blue]Parsing: {json_file_path}[/bold blue]"))
    
    total_queries = 0
    total_runs = 0
    successful_runs = 0
    queries_with_success = 0
    
    # Process each battery
    for battery in data.get('batteries', []):
        battery_name = battery.get('name', 'Unknown Battery')
        console.print(f"\n[bold cyan]Battery: {battery_name}[/bold cyan]")
        
        # Process each query in the battery
        for query_data in battery.get('queries', []):
            query_text = query_data.get('query', 'No query text')
            query_index = query_data.get('query_index', 'Unknown')
            
            total_queries += 1
            query_has_success = False
            
            # Display query with colored panel
            query_panel = Panel(
                Text(query_text, style="bold white"),
                title=f"[yellow]Query #{query_index}[/yellow]",
                border_style="yellow",
                padding=(0, 1)
            )
            console.print(query_panel)
            
            # Process each model's runs
            for model_data in query_data.get('models', []):
                model_name = model_data.get('model', 'Unknown Model')
                
                for run in model_data.get('runs', []):
                    total_runs += 1
                    success = run.get('success', False)
                    response = run.get('response', 'No response')
                    error = run.get('error')
                    
                    if success:
                        successful_runs += 1
                        query_has_success = True
                        # Display successful response
                        response_panel = Panel(
                            Text(response, style="white"),
                            title=f"[green]✓ Response ({model_name})[/green]",
                            border_style="green",
                            padding=(0, 1)
                        )
                        console.print(response_panel)
                    else:
                        # Display error
                        error_text = error if error else "Unknown error"
                        error_panel = Panel(
                            Text(error_text, style="red"),
                            title=f"[red]✗ Error ({model_name})[/red]",
                            border_style="red",
                            padding=(0, 1)
                        )
                        console.print(error_panel)
                    
                    # Extract and display tools used
                    tools_used = extract_tools_from_run(run)
                    if tools_used:
                        tools_text = " • ".join(tools_used)
                        console.print(f"[red]Tools: {tools_text}[/red]")
                    
                    # Add some spacing
                    console.print("")
            
            # Count queries that had at least one successful run
            if query_has_success:
                queries_with_success += 1
    
    # Display summary
    console.print(Rule("[bold blue]Summary[/bold blue]"))
    console.print(f"[cyan]Total queries processed: {total_queries}[/cyan]")
    console.print(f"[cyan]Total runs executed: {total_runs}[/cyan]")
    console.print(f"[green]Successful runs: {successful_runs}[/green]")
    console.print(f"[red]Failed runs: {total_runs - successful_runs}[/red]")
    console.print(f"[yellow]Queries with at least one success: {queries_with_success}[/yellow]")
    
    if total_runs > 0:
        run_success_rate = (successful_runs / total_runs) * 100
        console.print(f"[yellow]Run success rate: {run_success_rate:.1f}%[/yellow]")
    
    if total_queries > 0:
        query_success_rate = (queries_with_success / total_queries) * 100
        console.print(f"[yellow]Query success rate: {query_success_rate:.1f}%[/yellow]")

def main():
    """Main function to handle command line arguments."""
    
    console = Console()
    
    # Check if file path is provided
    if len(sys.argv) != 2:
        console.print("[red]Usage: python parse_responses.py <json_file_path>[/red]")
        console.print("[yellow]Example: python parse_responses.py output/20250722_161854.json[/yellow]")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    
    # Check if file exists
    if not Path(json_file_path).exists():
        console.print(f"[red]Error: File '{json_file_path}' does not exist.[/red]")
        sys.exit(1)
    
    # Parse and display
    parse_and_display_responses(json_file_path)

if __name__ == "__main__":
    main() 