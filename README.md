# MCP Overflow Experiment Suite

This experiment suite tests multiple LLM models with MCP (Model Context Protocol) servers to evaluate their performance across various queries.

## Features

- 🤖 **Multi-Model Testing**: Tests multiple AI models simultaneously
- 🔧 **MCP Server Integration**: Applies MCP servers during query processing
- 📊 **Streaming Output**: Real-time colored console output with timestamps
- 💾 **JSON Logging**: Comprehensive results saved to timestamped JSON files
- 🎨 **Colored Console**: Beautiful colored output showing experiment progress

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API Key**:
   ```bash
   cp env.example.py env.py
   ```
   and populate with secret values

## Usage

Run the complete experiment suite:

```bash
python main.py
```


1. Complete workflow:
```bash
  make run-battery BATTERY=test_pretrain_knowledge_with_no_mcp_servers
  make pipeline  # Automatically processes latest results → CSV
```
2. Step-by-step:
```bash 
  make run-battery BATTERY=test_pretrain_knowledge_with_no_mcp_servers
  make evaluate-latest
  make convert-latest-to-csv
```
3. Manual file specification:
```bash
  make evaluate FILE=output/20250805_123456.json
  make convert-to-csv FILE=output/evaluations/evaluated_20250805_123456.json OUTPUT=my_results.csv
```

## Output Files

Results are saved to `output/[timestamp].json` with the following structure:

```json
{
  "timestamp": "20240115_143025",
  "experiment_start": "2024-01-15T14:30:25.123456",
  "total_queries": 12,
  "results": [
    {
      "query": "What is the capital of France?",
      "model": "anthropic/claude-sonnet-4",
      "experiment": "browsing",
      "timestamp": "2024-01-15T14:30:25.123456",
      "mcp_servers": ["context7"],
      "success": true,
      "response": "The capital of France is Paris...",
      "error": null,
      "duration": 2.34
    }
  ]
}
```