#!/usr/bin/env python3
"""
Statistics Comparison Chart Generator

This script takes multiple experiment JSON output files and creates comparison charts
about all numeric statistics between runs. Assumes consistent schema between files.

Usage: python comparison_charts.py file1.json file2.json file3.json [--output output_dir]
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

# Set style for better looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class StatisticsComparator:
    def __init__(self, json_files: List[str], output_dir: str = "comparison_charts"):
        self.json_files = json_files
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.data = []
        self.run_level_metrics = []
        self.model_summary_metrics = []
        self.overall_metrics = []

    def load_json_files(self):
        """Load and parse all JSON files."""
        for file_path in self.json_files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    data["source_file"] = Path(file_path).stem
                    self.data.append(data)
                print(f"Loaded: {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    def extract_metrics(self):
        """Extract all numeric metrics from the loaded data."""
        for data in self.data:
            source_file = data["source_file"]
            timestamp = data.get("metadata", {}).get("timestamp", "unknown")

            # Process each battery
            for battery in data.get("batteries", []):
                battery_name = battery.get("name", "unknown")

                # Process each query
                for query in battery.get("queries", []):
                    query_text = (
                        query.get("query", "unknown")[:50] + "..."
                    )  # Truncate for display

                    # Process each model
                    for model_data in query.get("models", []):
                        model_name = model_data.get("model", "unknown")

                        # Extract run-level metrics
                        for run in model_data.get("runs", []):
                            metrics = run.get("metrics", {})
                            run_record = {
                                "source_file": source_file,
                                "timestamp": timestamp,
                                "battery": battery_name,
                                "query": query_text,
                                "model": model_name,
                                "success": run.get("success", False),
                                "evaluation": run.get("evaluation", "unknown"),
                                **{
                                    k: v
                                    for k, v in metrics.items()
                                    if isinstance(v, (int, float))
                                },
                            }
                            self.run_level_metrics.append(run_record)

                        # Extract model summary metrics
                        summary = model_data.get("summary", {})
                        if summary:
                            summary_record = {
                                "source_file": source_file,
                                "timestamp": timestamp,
                                "battery": battery_name,
                                "model": model_name,
                                **{
                                    k: v
                                    for k, v in summary.items()
                                    if isinstance(v, (int, float))
                                },
                            }
                            self.model_summary_metrics.append(summary_record)

                # Calculate overall battery metrics
                overall_metrics = self._calculate_overall_metrics(
                    battery, source_file, timestamp
                )
                self.overall_metrics.append(overall_metrics)

    def _calculate_overall_metrics(
        self, battery: Dict[str, Any], source_file: str, timestamp: str
    ) -> Dict[str, Any]:
        """Calculate overall metrics for a battery."""
        total_queries = len(battery.get("queries", []))
        total_models = len(battery.get("models", []))

        # Count totals across all runs
        total_runs = 0
        successful_runs = 0
        total_cost = 0
        total_latency = 0
        total_tools_used = 0
        total_input_tokens = 0
        total_output_tokens = 0
        run_count = 0

        for query in battery.get("queries", []):
            for model_data in query.get("models", []):
                runs = model_data.get("runs", [])
                total_runs += len(runs)
                for run in runs:
                    if run.get("success", False):
                        successful_runs += 1
                    metrics = run.get("metrics", {})
                    total_cost += metrics.get("cost", 0)
                    total_latency += metrics.get("latency", 0)
                    total_tools_used += metrics.get("tools_used_count", 0)
                    total_input_tokens += metrics.get("input_tokens", 0)
                    total_output_tokens += metrics.get("output_tokens", 0)
                    run_count += 1

        success_rate = (successful_runs / total_runs) if total_runs > 0 else 0
        avg_latency = (total_latency / run_count) if run_count > 0 else 0

        return {
            "source_file": source_file,
            "timestamp": timestamp,
            "battery": battery.get("name", "unknown"),
            "total_queries": total_queries,
            "total_models": total_models,
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "success_rate": success_rate,
            "total_cost": total_cost,
            "avg_latency": avg_latency,
            "total_tools_used": total_tools_used,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        }

    def create_cost_vs_success_chart(self):
        """Create chart showing relation between cost and success per model."""
        if not self.run_level_metrics:
            print("No run-level metrics found")
            return

        df = pd.DataFrame(self.run_level_metrics)
        
        if 'cost' not in df.columns:
            print("Cost data not found in metrics")
            return

        plt.figure(figsize=(12, 8))
        
        # Create scatter plot for each model
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            
            # Separate success and failure points
            success_data = model_data[model_data['success'] == True]
            failure_data = model_data[model_data['success'] == False]
            
            # Plot success points
            if len(success_data) > 0:
                plt.scatter(success_data['cost'], [1] * len(success_data), 
                           alpha=0.7, label=f'{model} - Success', marker='o', s=60)
            
            # Plot failure points
            if len(failure_data) > 0:
                plt.scatter(failure_data['cost'], [0] * len(failure_data), 
                           alpha=0.7, label=f'{model} - Failure', marker='x', s=60)

        plt.xlabel('Cost ($)')
        plt.ylabel('Success (1) / Failure (0)')
        plt.title('Cost vs Success Rate by Model')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / "cost_vs_success_by_model.png", 
                   dpi=300, bbox_inches="tight", pad_inches=0.4)
        plt.close()
        print("Created cost vs success chart")

    def create_tokens_vs_success_chart(self):
        """Create chart showing relation between token count and success per model."""
        if not self.run_level_metrics:
            print("No run-level metrics found")
            return

        df = pd.DataFrame(self.run_level_metrics)
        
        # Calculate total tokens
        if 'input_tokens' in df.columns and 'output_tokens' in df.columns:
            df['total_tokens'] = df['input_tokens'] + df['output_tokens']
        elif 'input_tokens' in df.columns:
            df['total_tokens'] = df['input_tokens']
        elif 'output_tokens' in df.columns:
            df['total_tokens'] = df['output_tokens']
        else:
            print("Token data not found in metrics")
            return

        plt.figure(figsize=(12, 8))
        
        # Create scatter plot for each model
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            
            # Separate success and failure points
            success_data = model_data[model_data['success'] == True]
            failure_data = model_data[model_data['success'] == False]
            
            # Plot success points
            if len(success_data) > 0:
                plt.scatter(success_data['total_tokens'], [1] * len(success_data), 
                           alpha=0.7, label=f'{model} - Success', marker='o', s=60)
            
            # Plot failure points
            if len(failure_data) > 0:
                plt.scatter(failure_data['total_tokens'], [0] * len(failure_data), 
                           alpha=0.7, label=f'{model} - Failure', marker='x', s=60)

        plt.xlabel('Total Tokens')
        plt.ylabel('Success (1) / Failure (0)')
        plt.title('Token Count vs Success Rate by Model')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / "tokens_vs_success_by_model.png", 
                   dpi=300, bbox_inches="tight", pad_inches=0.4)
        plt.close()
        print("Created tokens vs success chart")

    def create_latency_vs_success_chart(self):
        """Create chart showing relation between latency and success per model."""
        if not self.run_level_metrics:
            print("No run-level metrics found")
            return

        df = pd.DataFrame(self.run_level_metrics)
        
        if 'latency' not in df.columns:
            print("Latency data not found in metrics")
            return

        plt.figure(figsize=(12, 8))
        
        # Create scatter plot for each model
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            
            # Separate success and failure points
            success_data = model_data[model_data['success'] == True]
            failure_data = model_data[model_data['success'] == False]
            
            # Plot success points
            if len(success_data) > 0:
                plt.scatter(success_data['latency'], [1] * len(success_data), 
                           alpha=0.7, label=f'{model} - Success', marker='o', s=60)
            
            # Plot failure points
            if len(failure_data) > 0:
                plt.scatter(failure_data['latency'], [0] * len(failure_data), 
                           alpha=0.7, label=f'{model} - Failure', marker='x', s=60)

        plt.xlabel('Latency (seconds)')
        plt.ylabel('Success (1) / Failure (0)')
        plt.title('Latency vs Success Rate by Model')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / "latency_vs_success_by_model.png", 
                   dpi=300, bbox_inches="tight", pad_inches=0.4)
        plt.close()
        print("Created latency vs success chart")

    def create_tool_usage_success_comparison(self):
        """Create chart comparing success rates for tool usage scenarios (grouped vs ungrouped)."""
        if not self.overall_metrics:
            print("No overall metrics found")
            return

        df = pd.DataFrame(self.overall_metrics)
        
        # Filter files containing "tool_usage"
        tool_usage_files = df[df['source_file'].str.contains('tool_usage', case=False, na=False)]
        
        if tool_usage_files.empty:
            print("No files containing 'tool_usage' found")
            return

        # Categorize as grouped (_all_) or ungrouped (_only_)
        grouped_files = tool_usage_files[tool_usage_files['source_file'].str.contains('_all_', case=False, na=False)]
        ungrouped_files = tool_usage_files[tool_usage_files['source_file'].str.contains('_only_', case=False, na=False)]

        plt.figure(figsize=(10, 6))
        
        categories = []
        success_rates = []
        
        if not grouped_files.empty:
            avg_success_grouped = grouped_files['success_rate'].mean()
            categories.append('Grouped (_all_)')
            success_rates.append(avg_success_grouped)
        
        if not ungrouped_files.empty:
            avg_success_ungrouped = ungrouped_files['success_rate'].mean()
            categories.append('Ungrouped (_only_)')
            success_rates.append(avg_success_ungrouped)
        
        if categories:
            bars = plt.bar(categories, success_rates, color=['skyblue', 'lightcoral'])
            plt.xlabel('Tool Usage Scenario')
            plt.ylabel('Success Rate')
            plt.title('Success Rate Comparison: Tool Usage Grouped vs Ungrouped')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for i, (bar, rate) in enumerate(zip(bars, success_rates)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            plt.savefig(self.output_dir / "tool_usage_success_comparison.png", 
                       dpi=300, bbox_inches="tight", pad_inches=0.4)
            plt.close()
            print("Created tool usage success comparison chart")
        else:
            print("No valid tool usage comparison data found")

    def create_pretrain_knowledge_success_comparison(self):
        """Create chart comparing success rates for pretrain knowledge scenarios."""
        if not self.overall_metrics:
            print("No overall metrics found")
            return

        df = pd.DataFrame(self.overall_metrics)
        
        # Filter files containing pretrain knowledge patterns
        with_no_files = df[df['source_file'].str.contains('pretrain_knowledge_with_no', case=False, na=False)]
        with_all_files = df[df['source_file'].str.contains('pretrain_knowledge_with_all', case=False, na=False)]

        plt.figure(figsize=(10, 6))
        
        categories = []
        success_rates = []
        
        if not with_no_files.empty:
            avg_success_no = with_no_files['success_rate'].mean()
            categories.append('With No Knowledge')
            success_rates.append(avg_success_no)
        
        if not with_all_files.empty:
            avg_success_all = with_all_files['success_rate'].mean()
            categories.append('With All Knowledge')
            success_rates.append(avg_success_all)
        
        if categories:
            bars = plt.bar(categories, success_rates, color=['lightcoral', 'lightgreen'])
            plt.xlabel('Pretrain Knowledge Scenario')
            plt.ylabel('Success Rate')
            plt.title('Success Rate Comparison: Pretrain Knowledge With No vs With All')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for i, (bar, rate) in enumerate(zip(bars, success_rates)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            plt.savefig(self.output_dir / "pretrain_knowledge_success_comparison.png", 
                       dpi=300, bbox_inches="tight", pad_inches=0.4)
            plt.close()
            print("Created pretrain knowledge success comparison chart")
        else:
            print("No valid pretrain knowledge comparison data found")



    def generate_all_charts(self):
        """Generate specific comparison charts as requested."""
        print(f"Loading {len(self.json_files)} JSON files...")
        self.load_json_files()

        print("Extracting metrics...")
        self.extract_metrics()

        print(f"Found {len(self.run_level_metrics)} run-level records")
        print(f"Found {len(self.model_summary_metrics)} model summary records")
        print(f"Found {len(self.overall_metrics)} overall metric records")

        print("Creating cost vs success chart...")
        self.create_cost_vs_success_chart()

        print("Creating tokens vs success chart...")
        self.create_tokens_vs_success_chart()

        print("Creating latency vs success chart...")
        self.create_latency_vs_success_chart()

        print("Creating tool usage success comparison...")
        self.create_tool_usage_success_comparison()

        print("Creating pretrain knowledge success comparison...")
        self.create_pretrain_knowledge_success_comparison()

        print(f"\nAll requested charts saved to: {self.output_dir}")
        print(f"Generated 5 specific comparison charts from {len(self.json_files)} experiment files")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison charts from multiple experiment JSON files"
    )
    parser.add_argument("json_files", nargs="+", help="JSON files to compare")
    parser.add_argument(
        "--output",
        "-o",
        default="comparison_charts",
        help="Output directory for charts (default: comparison_charts)",
    )

    args = parser.parse_args()

    # Validate input files
    for file_path in args.json_files:
        if not Path(file_path).exists():
            print(f"Error: File {file_path} does not exist")
            sys.exit(1)

    comparator = StatisticsComparator(args.json_files, args.output)
    comparator.generate_all_charts()


if __name__ == "__main__":
    main()
