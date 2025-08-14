#!/usr/bin/env python3
"""
JSON to HTML Report Generator

This script converts experiment JSON output files into beautiful, scrollable HTML reports.
Usage: 
  Single file: python json_to_html.py <input_json_file> [output_html_file] [--per-model]
  Batch mode:  python json_to_html.py file1.json file2.json file3.json [--per-model]

Options:
  --per-model    Generate separate reports for each model instead of a unified report
  --output-dir   Directory to place output files (used in batch mode)
"""

import json
import sys
import os
import html
from datetime import datetime
from pathlib import Path
import argparse

def format_timestamp(timestamp_str):
    """Format timestamp string for display."""
    if timestamp_str is None:
        return "Unknown"
    try:
        if not isinstance(timestamp_str, str):
            return str(timestamp_str)
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return str(timestamp_str)

def format_duration(start_str, end_str):
    """Calculate and format duration between two timestamps."""
    if start_str is None or end_str is None:
        return "Unknown"
    try:
        start = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
        duration = end - start
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        elif minutes > 0:
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            return f"{int(seconds)}s"
    except:
        return "Unknown"

def get_status_color(evaluation):
    """Get color class for evaluation status."""
    if evaluation == "success":
        return "success"
    elif evaluation == "failure":
        return "failure"
    elif evaluation == "human_judge":
        return "warning"
    else:
        return "neutral"

def generate_html_report(data, output_file, target_model=None):
    """Generate HTML report from experiment data."""
    
    # Extract metadata
    metadata = data.get('metadata', {})
    batteries = data.get('batteries', [])
    
    if not batteries:
        print("No battery data found in JSON file")
        return
    
    battery = batteries[0]  # Assuming single battery for now
    experiment_name = battery.get('name', 'Unknown Experiment')
    models = battery.get('models', [])
    queries = battery.get('queries', [])
    
    # If target_model is specified, filter for that model only
    if target_model:
        experiment_name = f"{experiment_name} - {target_model}"
    
    # Calculate overall statistics
    total_queries = len(queries)
    if target_model:
        # Count queries that have results for the target model
        total_models = 1
        total_queries = 0
        for query in queries:
            for model_data in query.get('models', []):
                if model_data.get('model') == target_model:
                    total_queries += 1
                    break
    else:
        total_models = len(models)
    
    # Count successful runs across all queries and models
    total_runs = 0
    successful_runs = 0
    total_cost = 0
    total_latency = 0
    total_tools_used = 0
    total_tools_available = 0
    run_count = 0
    
    # Calculate total tools available from MCP servers
    mcp_servers = battery.get('mcp_servers', {})
    for server_name, server_info in mcp_servers.items():
        if isinstance(server_info, dict) and server_info.get('connected', False):
            tools = server_info.get('tools', [])
            if isinstance(tools, list):
                total_tools_available += len(tools)
    
    for query in queries:
        for model_data in query.get('models', []):
            # Skip this model if we're filtering for a specific model
            if target_model and model_data.get('model') != target_model:
                continue
                
            runs = model_data.get('runs', [])
            total_runs += len(runs)
            for run in runs:
                if run.get('evaluation') == 'success':
                    successful_runs += 1
                metrics = run.get('metrics', {})
                if 'cost' in metrics:
                    total_cost += metrics['cost']
                if 'latency' in metrics:
                    total_latency += metrics['latency']
                    run_count += 1
                if 'tools_used_count' in metrics:
                    total_tools_used += metrics['tools_used_count']
    
    success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
    avg_latency = (total_latency / run_count) if run_count > 0 else 0
    
    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Report: {html.escape(experiment_name)}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-label {{
            color: #666;
            text-transform: uppercase;
            font-size: 0.9em;
            letter-spacing: 1px;
        }}
        
        .section {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            overflow: hidden;
        }}
        
        .section-header {{
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .section-title {{
            font-size: 1.5em;
            font-weight: 600;
            color: #495057;
        }}
        
        .section-content {{
            padding: 20px;
        }}
        
        .query-item {{
            border: 1px solid #e9ecef;
            border-radius: 8px;
            margin-bottom: 20px;
            overflow: hidden;
        }}
        
        .query-header {{
            background: #f8f9fa;
            padding: 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .query-header:hover {{
            background: #e9ecef;
        }}
        
        .query-title {{
            font-weight: 600;
            color: #495057;
        }}
        
        .query-expected {{
            font-size: 0.9em;
            color: #6c757d;
            margin-top: 5px;
        }}
        
        .expand-icon {{
            font-size: 1.2em;
            transition: transform 0.3s ease;
        }}
        
        .query-content {{
            display: none;
            border-top: 1px solid #e9ecef;
        }}
        
        .query-content.expanded {{
            display: block;
        }}
        
        .query-header.expanded .expand-icon {{
            transform: rotate(180deg);
        }}
        
        .model-results {{
            display: grid;
            gap: 20px;
            padding: 20px;
        }}
        
        .model-section {{
            border: 1px solid #e9ecef;
            border-radius: 6px;
        }}
        
        .model-header {{
            background: #f8f9fa;
            padding: 15px;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .model-name {{
            color: #495057;
        }}
        
        .model-stats {{
            display: flex;
            gap: 15px;
            font-size: 0.9em;
            color: #6c757d;
        }}
        
        .runs-container {{
            padding: 15px;
        }}
        
        .run-item {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 10px;
        }}
        
        .run-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        
        .run-status {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .status-success {{
            background: #d4edda;
            color: #155724;
        }}
        
        .status-failure {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .status-warning {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .status-neutral {{
            background: #e2e3e5;
            color: #383d41;
        }}
        
        .run-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin-bottom: 10px;
            font-size: 0.9em;
        }}
        
        .metric {{
            text-align: center;
            padding: 8px;
            background: white;
            border-radius: 4px;
            border: 1px solid #e9ecef;
        }}
        
        .metric-value {{
            font-weight: bold;
            color: #495057;
        }}
        
        .metric-label {{
            color: #6c757d;
            font-size: 0.8em;
        }}
        
        .response-text {{
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 15px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            line-height: 1.4;
            max-height: 200px;
            overflow-y: auto;
        }}
        
        .toggle-btn {{
            background: #007bff;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9em;
        }}
        
        .toggle-btn:hover {{
            background: #0056b3;
        }}
        
        .search-box {{
            width: 100%;
            padding: 12px;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            font-size: 1em;
            margin-bottom: 20px;
        }}
        
        .no-results {{
            text-align: center;
            color: #6c757d;
            font-style: italic;
            padding: 40px;
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}
            
            .header {{
                padding: 20px;
            }}
            
            .header h1 {{
                font-size: 2em;
            }}
            
            .stats-grid {{
                grid-template-columns: 1fr;
            }}
            
            .run-metrics {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{html.escape(experiment_name)}</h1>
            <div class="subtitle">
                Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
                Duration: {format_duration(metadata.get('experiment_start') or '', metadata.get('experiment_end') or '')}
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{total_queries}</div>
                <div class="stat-label">Queries</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_models}</div>
                <div class="stat-label">Models</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_runs}</div>
                <div class="stat-label">Total Runs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{success_rate:.1f}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_latency:.2f}s</div>
                <div class="stat-label">Avg Latency</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${total_cost:.4f}</div>
                <div class="stat-label">Total Cost</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_tools_used}</div>
                <div class="stat-label">Tools Used</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_tools_available}</div>
                <div class="stat-label">Tools Available</div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-header">
                <div class="section-title">Query Results</div>
            </div>
            <div class="section-content">
                <input type="text" class="search-box" placeholder="Search queries..." id="searchBox">
                <div id="queryResults">
"""
    
    # Add queries
    for i, query in enumerate(queries):
        query_text = query.get('query', 'Unknown Query')
        expected = query.get('expected_outcome', 'Unknown')
        
        html_content += f"""
                    <div class="query-item" data-query="{html.escape(query_text.lower())}">
                        <div class="query-header" onclick="toggleQuery({i})">
                            <div>
                                <div class="query-title">Query {i + 1}: {html.escape(query_text)}</div>
                                <div class="query-expected">Expected: {html.escape(expected)}</div>
                            </div>
                            <div class="expand-icon">▼</div>
                        </div>
                        <div class="query-content" id="query-{i}">
                            <div class="model-results">
"""
        
        # Add model results for this query
        for model_data in query.get('models', []):
            model_name = model_data.get('model') or 'Unknown Model'
            
            # Skip this model if we're filtering for a specific model
            if target_model and model_name != target_model:
                continue
                
            summary = model_data.get('summary', {})
            eval_summary = model_data.get('evaluation_summary', {})
            runs = model_data.get('runs', [])
            
            success_rate = eval_summary.get('success_rate', 0)
            avg_latency = summary.get('avg_latency', 0)
            avg_cost = summary.get('avg_cost', 0)
            
            html_content += f"""
                                <div class="model-section">
                                    <div class="model-header">
                                        <div class="model-name">{html.escape(model_name)}</div>
                                        <div class="model-stats">
                                            <span>Success: {success_rate:.1f}%</span>
                                            <span>Latency: {avg_latency:.2f}s</span>
                                            <span>Cost: ${avg_cost:.4f}</span>
                                        </div>
                                    </div>
                                    <div class="runs-container">
"""
            
            # Add individual runs
            for j, run in enumerate(runs):
                evaluation = run.get('evaluation', 'unknown')
                status_class = f"status-{get_status_color(evaluation)}"
                metrics = run.get('metrics', {})
                response = run.get('response') or 'No response'
                timestamp = format_timestamp(run.get('timestamp') or '')
                
                html_content += f"""
                                        <div class="run-item">
                                            <div class="run-header">
                                                <span>Run {j + 1} - {timestamp}</span>
                                                <span class="run-status {status_class}">{evaluation}</span>
                                            </div>
                                            <div class="run-metrics">
                                                <div class="metric">
                                                    <div class="metric-value">{metrics.get('latency', 0):.2f}s</div>
                                                    <div class="metric-label">Latency</div>
                                                </div>
                                                <div class="metric">
                                                    <div class="metric-value">{metrics.get('accuracy', 0):.2f}</div>
                                                    <div class="metric-label">Accuracy</div>
                                                </div>
                                                <div class="metric">
                                                    <div class="metric-value">${metrics.get('cost', 0):.4f}</div>
                                                    <div class="metric-label">Cost</div>
                                                </div>
                                                <div class="metric">
                                                    <div class="metric-value">{metrics.get('input_tokens', 0)}</div>
                                                    <div class="metric-label">Input Tokens</div>
                                                </div>
                                                <div class="metric">
                                                    <div class="metric-value">{metrics.get('output_tokens', 0)}</div>
                                                    <div class="metric-label">Output Tokens</div>
                                                </div>
                                                <div class="metric">
                                                    <div class="metric-value">{metrics.get('tools_used_count', 0)}</div>
                                                    <div class="metric-label">Tools Used</div>
                                                </div>
                                            </div>
                                            <div class="response-text">{html.escape(response)}</div>
                                        </div>
"""
            
            html_content += """
                                    </div>
                                </div>
"""
        
        html_content += """
                            </div>
                        </div>
                    </div>
"""
    
    # Close HTML and add JavaScript
    html_content += f"""
                </div>
            </div>
        </div>
    </div>

    <script>
        function toggleQuery(index) {{
            const content = document.getElementById(`query-${{index}}`);
            const header = content.previousElementSibling;
            
            if (content.classList.contains('expanded')) {{
                content.classList.remove('expanded');
                header.classList.remove('expanded');
            }} else {{
                content.classList.add('expanded');
                header.classList.add('expanded');
            }}
        }}
        
        function filterQueries() {{
            const searchTerm = document.getElementById('searchBox').value.toLowerCase();
            const queryItems = document.querySelectorAll('.query-item');
            let visibleCount = 0;
            
            queryItems.forEach(item => {{
                const queryText = item.getAttribute('data-query');
                if (queryText.includes(searchTerm)) {{
                    item.style.display = 'block';
                    visibleCount++;
                }} else {{
                    item.style.display = 'none';
                }}
            }});
            
            // Show/hide no results message
            const resultsContainer = document.getElementById('queryResults');
            let noResultsMsg = document.getElementById('noResults');
            
            if (visibleCount === 0 && searchTerm.length > 0) {{
                if (!noResultsMsg) {{
                    noResultsMsg = document.createElement('div');
                    noResultsMsg.id = 'noResults';
                    noResultsMsg.className = 'no-results';
                    noResultsMsg.textContent = 'No queries match your search.';
                    resultsContainer.appendChild(noResultsMsg);
                }}
            }} else if (noResultsMsg) {{
                noResultsMsg.remove();
            }}
        }}
        
        // Add search functionality
        document.getElementById('searchBox').addEventListener('input', filterQueries);
        
        // Expand first query by default
        if (document.getElementById('query-0')) {{
            toggleQuery(0);
        }}
    </script>
</body>
</html>"""
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert JSON experiment results to HTML report')
    parser.add_argument('input_files', nargs='+', help='Input JSON file path(s)')
    parser.add_argument('output_file', nargs='?', help='Output HTML file path (only used for single file mode)')
    parser.add_argument('--per-model', action='store_true', 
                       help='Generate separate reports for each model')
    parser.add_argument('--output-dir', help='Output directory for batch processing (defaults to same directory as input files)')
    
    args = parser.parse_args()
    
    input_files = args.input_files
    is_batch_mode = len(input_files) > 1
    
    # Validate output directory for batch mode
    if is_batch_mode and args.output_file:
        print("Warning: output_file argument ignored in batch mode. Use --output-dir instead.")
    
    # Setup output directory
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Batch processing statistics
    total_files = len(input_files)
    successful_files = 0
    failed_files = 0
    total_reports_generated = 0
    
    print(f"Processing {total_files} file(s)...")
    
    for i, input_file in enumerate(input_files, 1):
        print(f"\n[{i}/{total_files}] Processing: {input_file}")
        
        if not os.path.exists(input_file):
            print(f"  ✗ Error: Input file '{input_file}' not found.")
            failed_files += 1
            continue
        
        try:
            # Load JSON data
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            input_path = Path(input_file)
            
            # Determine output location
            if output_dir:
                output_base = output_dir / input_path.stem
            elif args.output_file and not is_batch_mode:
                output_base = Path(args.output_file).with_suffix('')
            else:
                output_base = input_path.with_suffix('')
            
            if args.per_model:
                # Generate separate reports for each model
                batteries = data.get('batteries', [])
                if not batteries:
                    print("  ✗ No battery data found in JSON file")
                    failed_files += 1
                    continue
                
                battery = batteries[0]  # Assuming single battery for now
                
                # Extract all unique model names
                model_names = set()
                for query in battery.get('queries', []):
                    for model_data in query.get('models', []):
                        model_name = model_data.get('model') or 'Unknown Model'
                        model_names.add(model_name)
                
                if not model_names:
                    print("  ✗ No models found in the data")
                    failed_files += 1
                    continue
                
                # Generate report for each model
                file_reports_generated = 0
                for model_name in sorted(model_names):
                    # Ensure model_name is not None and create safe filename
                    safe_model_name = (model_name or 'Unknown_Model').replace('/', '_').replace(':', '_')
                    
                    # Create model-specific output filename
                    model_output_file = f"{output_base}_{safe_model_name}.html"
                    
                    generate_html_report(data, str(model_output_file), target_model=model_name)
                    file_reports_generated += 1
                
                print(f"  ✓ Generated {file_reports_generated} per-model reports")
                total_reports_generated += file_reports_generated
            else:
                # Generate unified HTML report
                output_file = f"{output_base}.html"
                generate_html_report(data, str(output_file))
                print(f"  ✓ Generated unified report: {output_file}")
                total_reports_generated += 1
            
            successful_files += 1
            
        except json.JSONDecodeError as e:
            print(f"  ✗ Error: Invalid JSON file. {e}")
            failed_files += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed_files += 1
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"BATCH PROCESSING SUMMARY")
    print(f"{'='*50}")
    print(f"Total files processed: {total_files}")
    print(f"Successful: {successful_files}")
    print(f"Failed: {failed_files}")
    print(f"Total reports generated: {total_reports_generated}")
    
    if failed_files > 0:
        print(f"\n⚠️  {failed_files} file(s) failed to process. Check the error messages above.")
        sys.exit(1)
    else:
        print(f"\n✅ All files processed successfully!")

if __name__ == "__main__":
    main() 