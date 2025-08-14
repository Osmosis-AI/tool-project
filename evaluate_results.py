#!/usr/bin/env python3

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import asyncio

# Import OpenAI/LLM for evaluation
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from env import OPENROUTER_KEY

console = Console()

@dataclass
class EvaluationResult:
    query: str
    predicted_answer: str
    correct_answer: str
    mcp_tools: List[str]
    confidence_score: float
    model: str
    experiment: str
    query_index: int
    run_index: int
    # Additional metrics from main.py
    latency: float
    accuracy: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tools_used_count: int
    # Success/failure status and cost
    success: bool
    estimated_cost: float

class ResultsEvaluator:
    def __init__(self, model_name: str = "anthropic/claude-sonnet-4", max_concurrent_evaluations: int = 10):
        """Initialize the evaluator with a specific model for evaluation"""
        self.llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=OPENROUTER_KEY,
            model_name=model_name,
            temperature=0.1  # Low temperature for consistent evaluation
        )
        self.logger = logging.getLogger("evaluator")
        
        # Add semaphore for controlling concurrent evaluations
        self.semaphore = asyncio.Semaphore(max_concurrent_evaluations)
        
    def calculate_estimated_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost based on model and token usage"""
        # Pricing per 1M tokens (approximate rates as of 2024)
        pricing = {
            # OpenAI models
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "gpt-5": {"input": 1.25, "output": 10.0},
            
            # Anthropic models
            "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
            "anthropic/claude-3-sonnet": {"input": 3.0, "output": 15.0},
            "anthropic/claude-3-opus": {"input": 15.0, "output": 75.0},
            "anthropic/claude-sonnet-4": {"input": 3.0, "output": 15.0},
            
            # Default fallback pricing
            "default": {"input": 1.0, "output": 3.0}
        }
        
        # Get pricing for the model or use default
        model_pricing = pricing.get(model, pricing["default"])
        
        # Calculate cost (pricing is per 1M tokens, so divide by 1,000,000)
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
        
        return input_cost + output_cost
        
    async def evaluate_single_response(
        self, 
        query: str, 
        predicted_answer: str, 
        correct_answer: str, 
        mcp_tools: List[str]
    ) -> float:
        """Evaluate a single response using the LLM with semaphore control"""
        async with self.semaphore:  # Control concurrent evaluations
        
            # Create the evaluation prompt
            evaluation_prompt = f"""For the following question: {query}

Please determine whether the predicted answer is correct. If the key information is answered, and related MCP tools are used, it is considered correct:

Predicted answer: {predicted_answer}
Correct answer: {correct_answer}
MCP tools used: {mcp_tools}

Only return a number from 0 to 1 with your confidence of the truth"""

        try:
            # Query the LLM for evaluation
            response = await self.llm.ainvoke([
                SystemMessage(content="You are an expert evaluator. Assess whether the predicted answer correctly addresses the question compared to the expected correct answer. Consider both content accuracy and tool usage appropriateness."),
                HumanMessage(content=evaluation_prompt)
            ])
            
            # Extract confidence score from response
            response_text = response.content.strip()
            
            # Try to extract a number between 0 and 1
            try:
                # Look for decimal numbers in the response
                import re
                numbers = re.findall(r'\b0?\.\d+\b|\b[01](?:\.\d+)?\b', response_text)
                if numbers:
                    score = float(numbers[0])
                    return max(0.0, min(1.0, score))  # Clamp between 0 and 1
                else:
                    # If no valid number found, try to parse the whole response
                    score = float(response_text)
                    return max(0.0, min(1.0, score))
            except ValueError:
                self.logger.warning(f"Could not parse evaluation score from: {response_text}")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error evaluating response: {e}")
            return 0.0
    
    async def evaluate_results_file(self, file_path: Path) -> Dict[str, Any]:
        """Evaluate all results in a JSON file"""
        console.print(f"[blue]Loading results from: {file_path}[/blue]")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            console.print(f"[red]Error loading file {file_path}: {e}[/red]")
            return {}
        
        evaluations = []
        total_runs = 0
        
        # Count total runs for progress tracking
        for battery in data.get('batteries', []):
            for query_data in battery.get('queries', []):
                for model_data in query_data.get('models', []):
                    total_runs += len(model_data.get('runs', []))
        
        console.print(f"[cyan]Found {total_runs} runs to evaluate[/cyan]")
        console.print(f"[blue]Using concurrent evaluation with max {self.semaphore._value} parallel requests[/blue]")
        
        # Collect all evaluation tasks first
        evaluation_tasks = []
        evaluation_metadata = []
        
        # Process each battery to collect evaluation tasks
        for battery in data.get('batteries', []):
            experiment_name = battery.get('name', 'unknown')
            
            # Process each query
            for query_data in battery.get('queries', []):
                query = query_data.get('query', '')
                expected_outcome = query_data.get('expected_outcome', '')
                query_index = query_data.get('query_index', 0)
                
                # Process each model
                for model_data in query_data.get('models', []):
                    model_name = model_data.get('model', '')
                    
                    # Process each run
                    for run_index, run in enumerate(model_data.get('runs', [])):
                        # Extract metrics from the run
                        metrics = run.get('metrics', {})
                        latency = metrics.get('latency', 0.0)
                        accuracy = metrics.get('accuracy', 0.0)
                        input_tokens = metrics.get('input_tokens', 0)
                        output_tokens = metrics.get('output_tokens', 0)
                        total_tokens = metrics.get('total_tokens', input_tokens + output_tokens)
                        tools_used_count = metrics.get('tools_used_count', 0)
                        mcp_tools = metrics.get('tools_used_list', [])
                        
                        # Calculate estimated cost for this run
                        estimated_cost = self.calculate_estimated_cost(
                            model_name, input_tokens, output_tokens
                        )
                        
                        success = run.get('success', False)
                        
                        # Store metadata for this evaluation
                        eval_metadata = {
                            'query': query,
                            'expected_outcome': expected_outcome,
                            'experiment_name': experiment_name,
                            'query_index': query_index,
                            'model_name': model_name,
                            'run_index': run_index,
                            'latency': latency,
                            'accuracy': accuracy,
                            'input_tokens': input_tokens,
                            'output_tokens': output_tokens,
                            'total_tokens': total_tokens,
                            'tools_used_count': tools_used_count,
                            'mcp_tools': mcp_tools,
                            'success': success,
                            'estimated_cost': estimated_cost
                        }
                        evaluation_metadata.append(eval_metadata)
                        
                        if success:
                            predicted_answer = run.get('response', '')
                            # Create evaluation task
                            task = self.evaluate_single_response(
                                query, predicted_answer, expected_outcome, mcp_tools
                            )
                            evaluation_tasks.append(task)
                        else:
                            # For failed runs, create a task that returns 0.0
                            evaluation_tasks.append(self._create_failed_evaluation_task())
        
        # Execute all evaluation tasks concurrently
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Evaluating {len(evaluation_tasks)} responses concurrently...", total=len(evaluation_tasks))
            
            # Process evaluations in batches to avoid overwhelming the API
            batch_size = min(50, len(evaluation_tasks))  # Process in batches of 50
            confidence_scores = []
            
            for i in range(0, len(evaluation_tasks), batch_size):
                batch = evaluation_tasks[i:i + batch_size]
                console.print(f"[blue]Processing evaluation batch {i//batch_size + 1}/{(len(evaluation_tasks) + batch_size - 1)//batch_size}[/blue]")
                
                try:
                    batch_results = await asyncio.gather(*batch, return_exceptions=True)
                    
                    # Handle results and exceptions
                    for result in batch_results:
                        if isinstance(result, Exception):
                            self.logger.warning(f"Evaluation failed: {result}")
                            confidence_scores.append(0.0)
                        else:
                            confidence_scores.append(result)
                        progress.advance(task)
                        
                except Exception as e:
                    self.logger.error(f"Batch evaluation failed: {e}")
                    # Fill with zeros for this batch
                    for _ in batch:
                        confidence_scores.append(0.0)
                        progress.advance(task)
        
        # Combine results with metadata
        for i, (metadata, confidence_score) in enumerate(zip(evaluation_metadata, confidence_scores)):
            evaluations.append(EvaluationResult(
                query=metadata['query'],
                predicted_answer=metadata.get('predicted_answer', ''),
                correct_answer=metadata['expected_outcome'],
                mcp_tools=metadata['mcp_tools'],
                confidence_score=confidence_score,
                model=metadata['model_name'],
                experiment=metadata['experiment_name'],
                query_index=metadata['query_index'],
                run_index=metadata['run_index'],
                latency=metadata['latency'],
                accuracy=metadata['accuracy'] if metadata['success'] else 0.0,
                input_tokens=metadata['input_tokens'],
                output_tokens=metadata['output_tokens'],
                total_tokens=metadata['total_tokens'],
                tools_used_count=metadata['tools_used_count'] if metadata['success'] else 0,
                success=metadata['success'],
                estimated_cost=metadata['estimated_cost']
            ))
        
        return {
            'metadata': {
                'evaluated_at': datetime.now().isoformat(),
                'total_evaluations': len(evaluations),
                'source_file': str(file_path),
                'evaluator_model': self.llm.model_name,
                'max_concurrent_evaluations': self.semaphore._value
            },
            'evaluations': [
                {
                    'query': eval_result.query,
                    'predicted_answer': eval_result.predicted_answer,
                    'correct_answer': eval_result.correct_answer,
                    'mcp_tools_used': eval_result.mcp_tools,
                    'confidence_score': eval_result.confidence_score,
                    'model': eval_result.model,
                    'experiment': eval_result.experiment,
                    'query_index': eval_result.query_index,
                    'run_index': eval_result.run_index,
                    'latency': eval_result.latency,
                    'accuracy': eval_result.accuracy,
                    'input_tokens': eval_result.input_tokens,
                    'output_tokens': eval_result.output_tokens,
                    'total_tokens': eval_result.total_tokens,
                    'tools_used_count': eval_result.tools_used_count,
                    'success': eval_result.success,
                    'estimated_cost': eval_result.estimated_cost
                }
                for eval_result in evaluations
            ],
            'summary': self._generate_summary(evaluations)
        }
    
    async def _create_failed_evaluation_task(self) -> float:
        """Create a task that returns 0.0 for failed evaluations"""
        return 0.0
    
    def _generate_summary(self, evaluations: List[EvaluationResult]) -> Dict[str, Any]:
        """Generate summary statistics from evaluations"""
        if not evaluations:
            return {}
        
        # Overall statistics
        total_evaluations = len(evaluations)
        avg_confidence = sum(e.confidence_score for e in evaluations) / total_evaluations
        high_confidence_count = sum(1 for e in evaluations if e.confidence_score >= 0.7)
        
        # Per-model statistics
        model_stats = {}
        for eval_result in evaluations:
            model = eval_result.model
            if model not in model_stats:
                model_stats[model] = []
            model_stats[model].append(eval_result.confidence_score)
        
        model_summary = {}
        for model, scores in model_stats.items():
            model_summary[model] = {
                'avg_confidence': sum(scores) / len(scores),
                'queries': len(scores),
                'high_confidence_runs': sum(1 for s in scores if s >= 0.7),
                'success_rate': sum(1 for s in scores if s >= 0.5) / len(scores)
            }
        
        # Per-experiment statistics
        experiment_stats = {}
        for eval_result in evaluations:
            experiment = eval_result.experiment
            if experiment not in experiment_stats:
                experiment_stats[experiment] = []
            experiment_stats[experiment].append(eval_result.confidence_score)
        
        experiment_summary = {}
        for experiment, scores in experiment_stats.items():
            experiment_summary[experiment] = {
                'avg_confidence': sum(scores) / len(scores),
                'queries': len(scores),
                'high_confidence_runs': sum(1 for s in scores if s >= 0.7),
                'success_rate': sum(1 for s in scores if s >= 0.5) / len(scores)
            }
        
        return {
            'overall': {
                'total_evaluations': total_evaluations,
                'average_confidence': avg_confidence,
                'high_confidence_count': high_confidence_count,
                'high_confidence_rate': high_confidence_count / total_evaluations,
                'success_rate': sum(1 for e in evaluations if e.confidence_score >= 0.5) / total_evaluations
            },
            'by_model': model_summary,
            'by_experiment': experiment_summary
        }

async def main():
    parser = argparse.ArgumentParser(description="Evaluate MCP experiment results with concurrent processing")
    parser.add_argument('input_files', nargs='+', help='JSON result files to evaluate')
    parser.add_argument('--output-dir', default='output/evaluations', help='Output directory for evaluation results')
    parser.add_argument('--evaluator-model', default='anthropic/claude-sonnet-4', help='Model to use for evaluation')
    parser.add_argument('--max-concurrent', type=int, default=10, help='Maximum concurrent evaluations (default: 10)')
    parser.add_argument('--parallel-files', action='store_true', help='Process multiple files in parallel')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = ResultsEvaluator(
        model_name=args.evaluator_model, 
        max_concurrent_evaluations=args.max_concurrent
    )
    
    console.print(f"[bold green]Starting evaluation of {len(args.input_files)} files[/bold green]")
    console.print(f"[blue]Max concurrent evaluations: {args.max_concurrent}[/blue]")
    console.print(f"[blue]Parallel file processing: {args.parallel_files}[/blue]")
    
    async def process_single_file(input_file):
        """Process a single file and return its results"""
        input_path = Path(input_file)
        if not input_path.exists():
            console.print(f"[red]File not found: {input_path}[/red]")
            return None
        
        console.print(f"\n[blue]Processing: {input_path.name}[/blue]")
        
        try:
            # Evaluate the file
            evaluation_results = await evaluator.evaluate_results_file(input_path)
            
            if not evaluation_results:
                return None
            
            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"evaluated_{timestamp}_{input_path.stem}.json"
            output_path = output_dir / output_filename
            
            # Save results
            with open(output_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
            
            console.print(f"[green]Saved evaluation results to: {output_path}[/green]")
            
            # Print summary
            summary = evaluation_results.get('summary', {})
            overall = summary.get('overall', {})
            
            console.print(f"\n[cyan]Summary for {input_path.name}:[/cyan]")
            console.print(f"  Total evaluations: {overall.get('total_evaluations', 0)}")
            console.print(f"  Average confidence: {overall.get('average_confidence', 0):.3f}")
            console.print(f"  Success rate (≥0.5): {overall.get('success_rate', 0):.1%}")
            console.print(f"  High confidence rate (≥0.7): {overall.get('high_confidence_rate', 0):.1%}")
            
            # Print per-model summary
            by_model = summary.get('by_model', {})
            if by_model:
                console.print(f"\n[yellow]Per-model performance:[/yellow]")
                for model, stats in by_model.items():
                    console.print(f"  {model}:")
                    console.print(f"    Success rate: {stats.get('success_rate', 0):.1%}")
                    console.print(f"    Avg confidence: {stats.get('avg_confidence', 0):.3f}")
            
            return output_path
            
        except Exception as e:
            console.print(f"[red]Error processing {input_path.name}: {e}[/red]")
            return None
    
    if args.parallel_files and len(args.input_files) > 1:
        # Process files in parallel
        console.print(f"[blue]Processing {len(args.input_files)} files in parallel...[/blue]")
        tasks = [process_single_file(input_file) for input_file in args.input_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for result in results if result is not None and not isinstance(result, Exception))
        console.print(f"[green]Successfully processed {success_count}/{len(args.input_files)} files[/green]")
    else:
        # Process files sequentially
        for input_file in args.input_files:
            await process_single_file(input_file)
    
    console.print(f"\n[bold green]Evaluation complete! Results saved to: {output_dir}[/bold green]")

if __name__ == "__main__":
    asyncio.run(main())