# JupyMCP

JupyMCP is a Model Context Protocol (MCP) for Jupyter Notebooks.

## Usage

```json
{
    "mcpServers": {
        "jupymcp": {
            "command": "uvx",
            "args": ["jupymcp",
                "--server-url", "http://localhost:8888",
                "--token", "<token>",
                "--path", "<path>"]
        }
    }
}
```

```python
from mcp.client.stdio import StdioServerParameters
from swarmx import Swarm

swarm = Swarm(
    jupyter=StdioServerParameters(
        command="uvx",
        args=["jupymcp", "--server-url", "http://localhost:8890", "--token", "MY_TOKEN", "--path", "main.ipynb"],
    )
)
```

## Alternatives

- [jupyter-mcp-server](https://github.com/datalayer/jupyter-mcp-server)
- [jupyter-mcp](https://pypi.org/project/jupyter-mcp/)
- [mcp-jupyter](https://pypi.org/project/mcp-jupyter/)

## Why yet another one?

I personally want a full-featured Jupyter Notebook server that can be used as a MCP server.
All of the above alternatives are either not meeting my requirements (e.g. lack of editing).

## Why not a folk of one of the above?

I think it's better to start from scratch with LLM assistance. LLM-driven bootstrap is fun.

## Roadmap

- [ ] Multiple Kernel support
- [ ] Multiple Notebook support
- [ ] Notebook import/export
