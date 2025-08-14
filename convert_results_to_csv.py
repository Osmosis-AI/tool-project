#!/usr/bin/env python3

import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import statistics

# Model pricing per 1M tokens (input, output) - approximations in USD
MODEL_PRICING = {
    "anthropic/claude-sonnet-4": (0.015, 0.075),
    "moonshotai/kimi-k2": (0.001, 0.002),
    "openai/o3": (0.060, 0.120),
    "google/gemini-2.5-pro": (0.00125, 0.005),
    "composite": (0.020, 0.050)  # Average estimate
}

def parse_evaluation_file(file_path: Path) -> Dict[str, Any]:
    """Parse a JSON evaluation file and return the data"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return {}

def extract_experiment_type(experiment_name: str) -> str:
    """Extract whether experiment uses MCP or not from experiment name"""
    # Only use "W/O MCP" and "MCP" pattern for pretrain knowledge experiments
    if experiment_name in ["test_pretrain_knowledge_with_no_mcp_servers", "test_pretrain_knowledge_with_all_mcp_servers"]:
        if "no_mcp" in experiment_name.lower():
            return "W/O MCP"
        else:
            return "MCP"
    else:
        # For other experiments, extract the part after "test_tool_usage_"
        if experiment_name.startswith("test_tool_usage_"):
            return experiment_name[len("test_tool_usage_"):]
        else:
            return experiment_name

def get_model_display_name(model_name: str) -> str:
    """Convert API model name to display name"""
    mapping = {
        "anthropic/claude-sonnet-4": "Claude sonnet 4",
        "moonshotai/kimi-k2": "Kimi K2", 
        "openai/o3": "Openai o3",
        "google/gemini-2.5-pro": "Gemini 2.5",
        "composite": "composite"
    }
    return mapping.get(model_name, model_name)

def calculate_metrics(evaluations: List[Dict], model: str, experiment: str) -> Dict[str, Any]:
    """Calculate metrics for a specific model and experiment combination"""
    
    # Filter evaluations for this model and experiment
    filtered_evals = [
        e for e in evaluations 
        if e.get('model') == model and e.get('experiment') == experiment
    ]
    
    if not filtered_evals:
        return {
            'queries': 0,
            'avg_tokens_in': 0,
            'avg_tokens_out': 0,
            'input_price_per_query': 0,
            'output_price_per_query': 0,
            'total_price_per_query': 0,
            'successful_runs': 0,
            'total_runs': 0,
            'success_rate': 0,
            'tools_available': 0,
            'tools_used': 0,
            'tool_usage_rate': 0,
            'average_latency': 0
        }
    
    # Count unique queries
    unique_queries = len(set(e.get('query', '') for e in filtered_evals))
    
    # Calculate token metrics from actual data
    token_in_values = [e.get('input_tokens', 0) for e in filtered_evals if e.get('input_tokens', 0) > 0]
    token_out_values = [e.get('output_tokens', 0) for e in filtered_evals if e.get('output_tokens', 0) > 0]
    
    avg_tokens_in = statistics.mean(token_in_values) if token_in_values else 0
    avg_tokens_out = statistics.mean(token_out_values) if token_out_values else 0
    
    # Calculate pricing based on actual token usage
    input_price, output_price = MODEL_PRICING.get(model, (0.01, 0.03))
    input_price_per_query = (avg_tokens_in / 1000000) * input_price
    output_price_per_query = (avg_tokens_out / 1000000) * output_price
    total_price_per_query = input_price_per_query + output_price_per_query
    
    # Calculate success metrics based on confidence score
    successful_runs = sum(1 for e in filtered_evals if e.get('confidence_score', 0) >= 0.5)
    total_runs = len(filtered_evals)
    success_rate = successful_runs / total_runs if total_runs > 0 else 0
    
    # Calculate tool usage metrics from actual data
    tools_used_counts = [e.get('tools_used_count', 0) for e in filtered_evals]
    tools_available = 20  # Approximate based on experiment_control.py
    
    tools_used = statistics.mean(tools_used_counts) if tools_used_counts else 0
    tool_usage_rate = tools_used / tools_available if tools_available > 0 else 0
    
    # Calculate average latency from actual data
    latency_values = [e.get('latency', 0) for e in filtered_evals if e.get('latency', 0) > 0]
    average_latency = statistics.mean(latency_values) if latency_values else 0
    
    return {
        'queries': unique_queries,
        'avg_tokens_in': avg_tokens_in,
        'avg_tokens_out': avg_tokens_out,
        'input_price_per_query': input_price_per_query,
        'output_price_per_query': output_price_per_query,
        'total_price_per_query': total_price_per_query,
        'successful_runs': successful_runs,
        'total_runs': total_runs,
        'success_rate': success_rate,
        'tools_available': tools_available,
        'tools_used': tools_used,
        'tool_usage_rate': tool_usage_rate,
        'average_latency': average_latency
    }

def convert_to_csv(json_files: List[Path], output_file: Path):
    """Convert evaluation JSON files to CSV format"""
    
    # Collect all data
    all_evaluations = []
    all_experiments = set()
    all_models = set()
    
    for json_file in json_files:
        data = parse_evaluation_file(json_file)
        if not data:
            continue
            
        evaluations = data.get('evaluations', [])
        all_evaluations.extend(evaluations)
        
        for eval_data in evaluations:
            all_experiments.add(eval_data.get('experiment', ''))
            all_models.add(eval_data.get('model', ''))
    
    # Group experiments by MCP vs no-MCP
    experiment_groups = defaultdict(list)
    for exp in all_experiments:
        exp_type = extract_experiment_type(exp)
        experiment_groups[exp_type].append(exp)
    
    # Prepare CSV data
    csv_data = []
    
    # Generate rows for each model
    for model in sorted(all_models):
        model_display = get_model_display_name(model)
        
        # For each experiment type 
        for exp_type in sorted(experiment_groups.keys()):
            if exp_type not in experiment_groups:
                # Create empty row if experiment type doesn't exist
                if exp_type in ["W/O MCP", "MCP"]:
                    # Use old pattern for pretrain knowledge experiments
                    row_name = f"{exp_type}: {model_display}"
                else:
                    # Use new pattern for tool usage experiments: model name + experiment suffix
                    row_name = f"{model_display} {exp_type}"
                csv_data.append([
                    row_name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ])
                continue
            
            # Find experiments of this type
            type_experiments = experiment_groups[exp_type]
            
            if not type_experiments:
                # Create empty row
                if exp_type in ["W/O MCP", "MCP"]:
                    # Use old pattern for pretrain knowledge experiments
                    row_name = f"{exp_type}: {model_display}"
                else:
                    # Use new pattern for tool usage experiments: model name + experiment suffix
                    row_name = f"{model_display} {exp_type}"
                csv_data.append([
                    row_name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ])
                continue
            
            # Aggregate metrics across all experiments of this type
            total_metrics = {
                'queries': 0,
                'avg_tokens_in': 0,
                'avg_tokens_out': 0,
                'input_price_per_query': 0,
                'output_price_per_query': 0,
                'total_price_per_query': 0,
                'successful_runs': 0,
                'total_runs': 0,
                'success_rate': 0,
                'tools_available': 0,
                'tools_used': 0,
                'tool_usage_rate': 0,
                'average_latency': 0
            }
            
            experiment_count = 0
            for experiment in type_experiments:
                metrics = calculate_metrics(all_evaluations, model, experiment)
                if metrics['total_runs'] > 0:  # Only count experiments with data
                    for key in total_metrics:
                        if key in ['success_rate', 'tool_usage_rate', 'average_latency']:
                            total_metrics[key] += metrics[key]
                        else:
                            total_metrics[key] += metrics[key]
                    experiment_count += 1
            
            # Average the rate metrics
            if experiment_count > 0:
                total_metrics['success_rate'] /= experiment_count
                total_metrics['tool_usage_rate'] /= experiment_count
                total_metrics['average_latency'] /= experiment_count
            
            # Create row name based on experiment type
            if exp_type in ["W/O MCP", "MCP"]:
                # Use old pattern for pretrain knowledge experiments
                row_name = f"{exp_type}: {model_display}"
            else:
                # Use new pattern for tool usage experiments: model name + experiment suffix
                row_name = f"{model_display} {exp_type}"
            csv_data.append([
                row_name,
                total_metrics['queries'],
                total_metrics['avg_tokens_in'],
                total_metrics['avg_tokens_out'],
                f"{total_metrics['input_price_per_query']:.6f}",
                f"{total_metrics['output_price_per_query']:.6f}",
                f"{total_metrics['total_price_per_query']:.6f}",
                total_metrics['successful_runs'],
                total_metrics['total_runs'],
                f"{total_metrics['success_rate']:.3f}",
                total_metrics['tools_available'],
                f"{total_metrics['tools_used']:.1f}",
                f"{total_metrics['tool_usage_rate']:.3f}",
                f"{total_metrics['average_latency']:.2f}"
            ])
    
    # Write CSV file
    headers = [
        "Model",
        "Queries",
        "Average Tokens In Per Query",
        "Average Tokens Out Per Query",
        "Input price/query (USD)",
        "Output price/query (USD)", 
        "Total price/query (USD)",
        "Successful runs",
        "Total Runs",
        "Success Rate",
        "Tools available",
        "Tools used", 
        "Tool usage rate",
        "Average Latency"
    ]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(csv_data)
    
    print(f"CSV file created: {output_file}")
    print(f"Generated {len(csv_data)} rows for {len(all_models)} models")

def main():
    parser = argparse.ArgumentParser(description="Convert evaluation results to CSV")
    parser.add_argument('input_files', nargs='+', help='JSON evaluation files to convert')
    parser.add_argument('--output', '-o', default='evaluation_results.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    # Convert input files to Path objects
    json_files = [Path(f) for f in args.input_files]
    output_file = Path(args.output)
    
    # Validate input files exist
    for json_file in json_files:
        if not json_file.exists():
            print(f"Error: File not found: {json_file}")
            return 1
    
    convert_to_csv(json_files, output_file)
    return 0

if __name__ == "__main__":
    exit(main())