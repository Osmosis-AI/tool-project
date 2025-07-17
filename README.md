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
   export OPENROUTER_KEY='your-openrouter-api-key-here'
   ```

## Usage

Run the complete experiment suite:

```bash
python main.py
```

## Console Output Format

The console displays colored output in the format:
```
[exp_name] - [model_name] - [date] - [streamed_output_line]
```

Example:
```
[browsing] - [anthropic/claude-sonnet-4] - [2024-01-15 14:30:25] - Stream: The capital of France is Paris...
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

## Experiment Configuration

Edit `experiment/experiment_control.py` to modify:

- **Models**: Add/remove models from `candidate_models`
- **Queries**: Modify the `queries` array
- **MCP Servers**: Configure MCP servers in the `mcp_servers` section

## Current Experiment: "browsing"

- **Models**: moonshotai/kimi-k2, anthropic/claude-sonnet-4, openai/o3
- **MCP Servers**: context7 (via @upstash/context7-mcp)
- **Queries**: 4 test queries covering various topics

## Requirements

- Python 3.7+
- OpenRouter API key
- Node.js (for MCP servers)

## File Structure

```
osmosis-mcpoverflow/
├── main.py                     # Entry point
├── experiment/
│   └── experiment_control.py   # Experiment logic and configuration
├── output/                     # JSON result files
├── requirements.txt            # Python dependencies
└── README.md                   # This file
``` # tool-project
