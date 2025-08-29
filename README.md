# JupyMCP

JupyMCP is a Model Context Protocol (MCP) server for Jupyter Notebooks that provides both tools and resources for comprehensive notebook interaction.

## Usage

```json
{
    "mcpServers": {
        "jupymcp": {
            "command": "uvx",
            "args": ["jupymcp"]
        }
    }
}
```

```python
from mcp.client.stdio import StdioServerParameters
from swarmx import Agent

agent = Agent(
    mcpServers={
        "jupymcp": StdioServerParameters(
            command="uvx",
            args=["jupymcp"],
        )
    }
)
```

You do not need to specify the server URL, token, or path. JupyMCP will automatically manage them for you.

## Alternatives

- [jupyter-mcp-server](https://github.com/datalayer/jupyter-mcp-server)
- [jupyter-mcp](https://pypi.org/project/jupyter-mcp/)
- [mcp-jupyter](https://pypi.org/project/mcp-jupyter/)

## Features

JupyMCP provides comprehensive Jupyter Notebook integration through MCP:

### Tools
- **Cell Execution**: Execute, add, and modify notebook cells, and return outputs (support image & audio)
- **Cell Management**: Add, insert, and manage code, markdown, and raw cells
- **Metadata Operations**: Get and set cell and notebook metadata

### Resources
- **Kernel Specs**: List available kernel specifications via `jupyter://kernelspecs`
- **Kernels**: List running kernels via `jupyter://kernels`
- **Notebooks**: Read and write notebook files via `notebook://{path}`

## Why yet another one?

I personally want a full-featured Jupyter Notebook server that can be used as a MCP server.
All of the above alternatives are either not meeting my requirements (e.g. lack of editing).

## Why not a folk of one of the above?

I think it's better to start from scratch with LLM assistance. LLM-driven bootstrap is fun.

## Roadmap

- [x] Multiple Kernel support
- [x] Multiple Notebook support
- [x] Multimedia output support
- [ ] Authentication & security
- [ ] Notebook import/export
