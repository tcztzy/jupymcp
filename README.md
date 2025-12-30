# JupyMCP

[![PyPI](https://img.shields.io/pypi/v/jupymcp?color=blue)](https://pypi.org/project/jupymcp/)
[![Type checked with ty](https://img.shields.io/badge/type%20checked-ty-304FFE?labelColor=555555)](https://github.com/astral-sh/ty)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-30173D?labelColor=555555)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

JupyMCP is a Model Context Protocol (MCP) server for Jupyter Notebooks that provides both tools and resources for comprehensive notebook interaction.

## Installation

Install JupyMCP using uvx (recommended) or pip:

```bash
# Using uvx (no installation needed, auto-managed)
uvx jupymcp

# Or using uv
uv tool install jupymcp

# Or using pip
pip install jupymcp
```

### Setting up Kernels

JupyMCP requires Jupyter kernels to execute code. Below are examples for setting up common kernels with isolated environments.

### Before installation

It is highly recommended that set `JUPYTER_PATH` if you want all kernels install to the managed environment.

#### Python (IPython)
```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install IPython kernel
pip install ipykernel
python -m ipykernel install --user --name myproject --display-name "Python (myproject)"
```

#### R
```r
# In R console - create renv environment
# if you are using macOS & Homebrew, please remember that set ~/.R/Makevars
install.packages("renv")
renv::init()

# Install IRkernel
renv::install('IRkernel')
IRkernel::installspec(name = 'myproject-r', displayname = 'R (myproject)')
```

#### Rust
```bash
# Install evcxr_jupyter
# it is recommended that using msvc toolchain on Windows to avoid dlltool.exe problem
cargo install evcxr_jupyter --root ${PWD}
evcxr_jupyter --install
```

#### Julia
```julia
# In Julia REPL - activate project environment
using Pkg
Pkg.activate(".")

# Install IJulia
Pkg.add("IJulia")

# Optionally install kernel with custom name
using IJulia
installkernel("Julia myproject")
```

To verify installed kernels:
```bash
jupyter kernelspec list
```

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
