import argparse
import asyncio
import logging

from jupyter_kernel_client import KernelClient
from jupyter_nbmodel_client import (
    NbModelClient,
    get_jupyter_notebook_websocket_url,
)
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("jupyter")

logger = logging.getLogger(__name__)

kernel: KernelClient
notebook: NbModelClient


def extract_output(output: dict) -> str:
    """
    Extracts readable output from a Jupyter cell output dictionary.

    Args:
        output (dict): The output dictionary from a Jupyter cell.

    Returns:
        str: A string representation of the output.
    """
    output_type = output.get("output_type")
    match output_type:
        case "stream":
            return output.get("text", "")
    if output_type == "stream":
        return output.get("text", "")
    elif output_type in ["display_data", "execute_result"]:
        data = output.get("data", {})
        if "text/plain" in data:
            return data["text/plain"]
        elif "text/html" in data:
            return "[HTML Output]"
        elif "image/png" in data:
            return "[Image Output (PNG)]"
        else:
            return f"[{output_type} Data: keys={list(data.keys())}]"
    elif output_type == "error":
        return output["traceback"]
    else:
        return f"[Unknown output type: {output_type}]"


@mcp.tool()
async def execute_cell(
    index: int,
) -> list[str]:
    """Execute a notebook cell by index.
    
    Args:
        index: Cell index to execute
        
    Returns:
        list[str]: List of outputs from the executed cell
    """
    result = notebook.execute_cell(index, kernel)
    return [extract_output(output) for output in result["outputs"]]


@mcp.tool()
async def add_and_execute_code_cell(source: str) -> list[str]:
    """Add and execute a code cell in a Jupyter notebook.

    Args:
        source: Code content

    Returns:
        list[str]: List of outputs from the executed cell
    """

    index = notebook.add_code_cell(source)
    return await execute_cell(index)


@mcp.tool()
async def set_and_execute_code_cell(index: int, source: str) -> list[str]:
    """Set and execute a code cell in a Jupyter notebook.

    Args:
        index: Cell index to set
        source: New cell source

    Returns:
        list[str]: List of outputs from the executed cell
    """
    notebook.set_cell_source(index, source)
    return await execute_cell(index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    websocket_url = get_jupyter_notebook_websocket_url(server_url=args.server_url, token=args.token, path=args.path)
    async def amain():
        global kernel
        global notebook
        with KernelClient(server_url=args.server_url, token=args.token) as kernel:
            async with NbModelClient(websocket_url) as notebook:
                for cell_type in ("code", "markdown", "raw"):
                    add_name = f"add_{cell_type}_cell"
                    add_func = getattr(notebook, add_name)
                    insert_name = f"insert_{cell_type}_cell"
                    insert_func = getattr(notebook, insert_name, None)
                    def add_cell(source: str, kwargs: dict | None = None) -> int:
                        return add_func(source, **(kwargs or {}))
                    def insert_cell(index: int, source: str, kwargs: dict | None = None):
                        insert_func(index, source, **(kwargs or {}))
                        return f"{cell_type.capitalize()} cell inserted."
                    mcp.tool(f"add_{cell_type}_cell", add_func.__doc__)(add_cell)
                    if insert_func is not None:
                        mcp.tool(f"insert_{cell_type}_cell", insert_func.__doc__)(insert_cell)
                mcp.tool()(notebook.get_cell_metadata)
                mcp.tool()(notebook.get_cell_source)
                mcp.tool()(notebook.get_notebook_metadata)
                mcp.tool()(notebook.set_cell_metadata)
                mcp.tool()(notebook.set_cell_source)
                mcp.tool()(notebook.set_notebook_metadata)
                await mcp.run_stdio_async()
    asyncio.run(amain())


if __name__ == "__main__":
    main()
