test_batteries = [
    {
        "name": "browsing",
        "mode": "experiment",
        "candidate_models": [
            "openai/o3-mini"
        ],
        "mcp_servers": {
            "context7": {
                "command": "npx",
                "args": ["-y", "@upstash/context7-mcp"]
            }
        },
        "queries": [
            "How do I query postgres in python?",
            # "how do I query redis in python?",
        ]
    }
]