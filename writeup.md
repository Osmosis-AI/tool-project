https://www.notion.so/gulp-ai/MCP-Evaluation-23f0eabf24548009b9cbddef46ebd35c


# MCP 3 way benchmark

    In this experiment, we set out to compare 3 popular LLMS and their interoperative capability with MCP servers: Moonshot AI's Kimi K2,Anthropic's Claude sonnet 4, and Openai's o3.

MCP is a mechanism which was developed to allow models to run external tools, so long as they are capable of structuring their outputs. The assumed demand for MCP in a business setting is steadily rising due to trends in the adoption of agents in both a primary and meta- business activity capacity. A company might for example have a model connected to its slack, notion, google drive, and linear kanban board and use it to seek information about the organization. They might also have a model that connects via MCP for its programming operations to increase the factuality of its queries, and finally, in their product, might use a model as a primary interactive function to perform actions, rather than a GUI interface for example, they might provide a text query field that spawns relevant charts. All of these functions require the model to be able to navigate query misses(when its queries might fail at some point in the MCP pipeline either due to its failure to structure its query or an issue with the underlying system it accesses), reason around tasks to be completed, what tools do what, all while not getting overwhelmed by the quantity of tools presented.
As it stands, the quantity of MCP tools on the market is quite sparse, in assembling this experiment it was evident this is still a developing ecosystem, still largely supported by volunteer implementations, however in the future it may be the case that certain traditionally human-operated multimodal tools e.g. in the field of CAD or music design may be designed for a MCP-first environment. Our experiment sought to explore the potential of these new tools in four quadrants based on these emerging trends: browsing, information gathering, programming, and visual reasoning. A final run was also performed with all tools available to assess each model's ability to decide which to use without being "overwhelmed". For each of the questions in these quadrants, we evaluated latency, accuracy, cost and quantity/name of each tool used. Each model was also given 3 tries per question to better average their strengths.

- pretrain knowledge: what are the colors of apple - simple knowledge (no tools vs tools)
- all tasks should be equally hard and multi-step, the number of tools increasing is the difficulty threshold
  - we're not trying to break the models, we care the most about how long they take
- smithery, the gateway we used to have multiple mcp servers, did not work well and had to be replaced. Using such gatweays should be considered a last resort because of this.

- 3390 total tools

- interesting errors

  - CURSOR: A Exceeding total tools limit
    You have 57 tools from enabled servers. Too many tools can degrade performance, and some models may not respect more than 40 tools.
  - ANTHROPIC: "{'message': 'Provider returned error', 'code': 400, 'metadata': {'raw': '{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"tools: Tool names must be unique.\"}}', 'provider_name': 'Anthropic'}}",
  - Directory listing: https://github.com/mark3labs/mcp-filesystem-server
    should have listed the directory, instead tried to jump ahead and got errors like

    ```
    [16:13:23] INFO     HTTP Request: POST https://openrouter.ai/api/v1/chat/completions "HTTP/1.1 200 OK"              _client.py:1740

    Invoking: `read_file` with `{'path': 'test_files/7805912b-c8da-4134-9b54-b590f884352a.csv'}`


    [16:13:31] ERROR    Run failed: Error: access denied - path outside allowed directories:                                main.py:647
                        /Users/jake/Desktop/osmosisai/osmosis-mcp-experiment/test_files/7805912b-c8da-4134-9b54-b590f884352a.csv
      ✗ Attempt 1: Failed - Error: access denied - path outside allowed directories: /Users/jake/Desktop/osmosisai/osmosis-mcp-experiment/tes
    [16:13:32] INFO     Starting run 2/3 for model openai/o3                                                                main.py:618
              INFO     Query: Please take the data from 'test_files/7805912b-c8da-4134-9b54-b590f884352a.csv' and create a main.py:619
                        pie chart showing the precipitation distribution in the different boroughs of New York. Then, tell
                        me which borough has the smallest share of precipitation.
    Using agent executor with 226 tools
    ```

  -
https://www.dbreunig.com/2025/07/30/how-kimi-was-post-trained-for-tool-use.html






please alter our recharts in @src/App.tsx to have the following bar charts:
- latencies:
    Model, Latency(ms)
    All tools: Claude 4,	34
    Relevant Tools: Claude 4,	28
    All tools: Kimi K2,	23
    Relevant Tools: Kimi K2,	26
    All tools: o3,	41
    Relevant Tools: o3,	36
    All tools: Composite,	32
    Relevant Tools: Composite,	30
- tool usage:
    Model, Tool usage(%)
    All tools: Claude 4,	12.33%
    Relevant Tools: Claude 4,	16.56%
    All tools: Kimi K2,	13.44%
    Relevant Tools: Kimi K2,	21.55%
    All tools: o3,	12.89%
    Relevant Tools: o3,	13.28%
    All tools: Composite,	15.20%
    Relevant Tools: Composite,	21.88%
- tokens out per query:
    Model, Tokens
    All tools: Claude 4,	13025
    Relevant Tools: Claude 4,	12284
    All tools: Kimi K2,	3030
    Relevant Tools: Kimi K2,	2907
    All tools: o3,	13834
    Relevant Tools: o3,	10614
    All tools: Composite,	9963
    Relevant Tools: Composite,	8602
- cost/query:
    - Cost/Query USD (Browsing)
        Model, Cost (USD)
        All tools: Claude 4,	$1.6024
        Relevant Tools: Claude 4,	$0.4075
        All tools: Kimi K2,	$0.0005
        Relevant Tools: Kimi K2,	$0.0005
        All tools: o3,	$1.6915
        Relevant Tools: o3,	$0.1339
        All tools: Composite,	$0.7939
        Relevant Tools: Composite,	$0.1272
    - Cost/Query USD (Info Gathering)
        Model, Cost (USD)
        All tools: Claude 4,	$7.2427
        Relevant Tools: Claude 4,	$7.3870
        All tools: Kimi K2,	$0.0002
        Relevant Tools: Kimi K2,	$0.0002
        All tools: o3,	$5.4285
        Relevant Tools: o3,	$4.7515
        All tools: Composite,	$2.9042
        Relevant Tools: Composite,	$2.7388
    - Cost/Query USD (Programming)
        Model, Cost (USD)
        All tools: Claude 4,	$2.8021
        Relevant Tools: Claude 4,	$2.3958
        All tools: Kimi K2,	$0.0002
        Relevant Tools: Kimi K2,	$0.0002
        All tools: o3,	$3.3387
        Relevant Tools: o3,	$1.9449
        All tools: Composite,	$1.4791
        Relevant Tools: Composite,	$1.0080
    - Cost/Query USD (Visual)
        Model, Cost (USD)
        All tools: Claude 4,	$7.02
        Relevant Tools: Claude 4,	$1.44
        All tools: Kimi K2,	$0.00
        Relevant Tools: Kimi K2,	$0.00
        All tools: o3,	$5.03
        Relevant Tools: o3,	$0.88
        All tools: Composite,	$2.75
        Relevant Tools: Composite,	$0.53
- success rate:
  - Success Rate (Browsing)
    	Model, Success (%)
      All tools: Claude 4,	20.0%
      Relevant Tools: Claude 4,	10.0%
      All tools: Kimi K2,	80.0%
      Relevant Tools: Kimi K2,	83.3%
      All tools: o3,	40.0%
      Relevant Tools: o3,	16.7%
      All tools: Composite,	46.7%
      Relevant Tools: Composite,	36.7%
  - Success Rate (Info Gathering)
    	Model, Success (%)
      All tools: Claude 4,	56.0%
      Relevant Tools: Claude 4,	41.2%
      All tools: Kimi K2,	42.0%
      Relevant Tools: Kimi K2,	42.9%
      All tools: o3,	45.7%
      Relevant Tools: o3,	42.1%
      All tools: Composite,	41.9%
      Relevant Tools: Composite,	42.1%
  - Success Rate (Programming)
    	Model, Success (%)
      All tools: Claude 4,	21.8%
      Relevant Tools: Claude 4,	38.2%
      All tools: Kimi K2,	27.3%
      Relevant Tools: Kimi K2,	34.5%
      All tools: o3,	20.0%
      Relevant Tools: o3,	36.4%
      All tools: Composite,	23.0%
      Relevant Tools: Composite,	36.4%
  - Success Rate (Visual)
      Model, Success (%)
      All tools: Claude 4,	53.3%
      Relevant Tools: Claude 4,	46.7%
      All tools: Kimi K2,	48.9%
      Relevant Tools: Kimi K2,	37.8%
      All tools: o3,	68.9%
      Relevant Tools: o3,	53.3%
      All tools: Composite,	57.0%
      Relevant Tools: Composite,	45.9%



in the section called "Pretrain Knowledge Evaluations" add graphs for:
- Success Rate
    Model, Success Rate
    W/O MCP: Claude sonnet 4, 85.33%
    MCP: Claude sonnet 4, 77.07%
    W/O MCP: Kimi K2, 84.53%
    MCP: Kimi K2, 85.33%
    W/O MCP: Openai o3, 85.07%
    MCP: Openai o3, 80.27%
    W/O MCP: composite, 84.98%
    MCP: composite, 80.89%
- Tokens per query
    Model, tokens out per query
    W/O MCP: Claude sonnet 4, 15
    MCP: Claude sonnet 4, 154
    W/O MCP: Kimi K2, 6
    MCP: Kimi K2, 6
    W/O MCP: Openai o3, 30
    MCP: Openai o3, 59
    W/O MCP: composite, 6
    MCP: composite, 24
- Latency (ms)
    Model, Latency (ms)
    W/O MCP: Claude sonnet 4, 6
    MCP: Claude sonnet 4, 15
    W/O MCP: Kimi K2, 5
    MCP: Kimi K2, 4
    W/O MCP: Openai o3, 13
    MCP: Openai o3, 16
    W/O MCP: composite, 8
    MCP: composite, 12
- Tool usage rate
    Model, Tool usage rate
    W/O MCP: Claude sonnet 4, 35.69%
    MCP: Claude sonnet 4, 80.84%
    W/O MCP: Kimi K2, 72.05%
    MCP: Kimi K2, 18.43%
- Price per query
    Model, Price per query
    W/O MCP: Claude sonnet 4, $0.0174
    MCP: Claude sonnet 4, $5.8167
    W/O MCP: Kimi K2, $0.0001
    MCP: Kimi K2, $0.0001
    W/O MCP: Openai o3, $0.0185
    MCP: Openai o3, $3.2457
    W/O MCP: composite, $0.0101
    MCP: composite, $2.0132
