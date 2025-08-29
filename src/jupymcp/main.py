import asyncio
import re
import sys
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

from jupyter_client import AsyncKernelManager, AsyncMultiKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import (
    AudioContent,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    TextContent,
)
from nbformat.corpus.words import generate_corpus_id as random_cell_id

from .model import (
    CodeCell,
    DisplayData,
    Error,
    ExecuteResult,
    JupyterNotebook,
    MarkdownCell,
    OutputMetadata,
    RawCell,
    Stream,
)

Output = ExecuteResult | DisplayData | Stream | Error
Cell = RawCell | MarkdownCell | CodeCell


class NotebookLockManager:
    """Manages file-based locks for notebook files to prevent concurrent modifications."""

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self._lock_creation_lock = asyncio.Lock()

    async def get_lock(self, notebook_path: Path) -> asyncio.Lock:
        """Get or create a lock for the given notebook path."""
        path_str = str(notebook_path.resolve())

        # Use a lock to prevent race conditions when creating new locks
        async with self._lock_creation_lock:
            if path_str not in self._locks:
                self._locks[path_str] = asyncio.Lock()
            return self._locks[path_str]

    @asynccontextmanager
    async def acquire_lock(self, notebook_path: Path):
        """Context manager to acquire and release a lock for a notebook file."""
        lock = await self.get_lock(notebook_path)
        async with lock:
            yield


# Global lock manager instance
_notebook_lock_manager = NotebookLockManager()


@asynccontextmanager
async def kernel_manager():
    try:
        km = cast(AsyncMultiKernelManager, AsyncMultiKernelManager())
        km.kernel_spec_manager = KernelSpecManager()
        yield km
    finally:
        await km.shutdown_all(now=True)


def _output_hook(cell: CodeCell, message: dict[str, Any]):  # noqa: C901
    """Callback on messages captured during a code snippet execution.

    The return list of updated output will be empty if no output where changed.
    It will equal all indexes if the outputs was cleared.

    Example:
        This callback is meant to be used with ``KernelClient.execute_interactive``::

            from functools from partial
            from jupyter_kernel_client import KernelClient
            with KernelClient(server_url, token) as kernel:
                outputs = []
                kernel.execute_interactive(
                    "print('hello')",
                    output_hook=partial(output_hook, outputs)
                )
                print(outputs)

    Args:
        outputs: List in which to append the output
        message: A kernel message

    Returns:
        list of output indexed updated
    """
    msg_type = message["header"]["msg_type"]
    content = message["content"]

    output = None
    match msg_type:
        case "execute_result":
            output = ExecuteResult(
                output_type="execute_result",
                metadata=content.get("metadata"),
                data=content.get("data"),
                execution_count=content.get("execution_count"),
            )
        case "stream":
            output = Stream(
                output_type="stream",
                name=content.get("name"),
                text=content.get("text"),
            )
        case "display_data":
            output = DisplayData(
                output_type="display_data",
                metadata=content.get("metadata")
                | {"transient": content.get("transient")},
                data=content.get("data"),
            )
        case "error":
            output = Error(
                output_type="error",
                ename=content.get("ename"),
                evalue=content.get("evalue"),
                traceback=content.get("traceback"),
            )
        case "clear_output":
            cell.outputs.clear()
        case "update_display_data":
            display_id = content.get("transient", {}).get("display_id")
            if display_id:
                for output in cell.outputs:
                    if not isinstance(output, DisplayData):
                        continue
                    if (
                        getattr(output.metadata, "transient", {}).get("display_id")
                        == display_id
                    ):
                        output.data = content.get("data")
                        output.metadata = OutputMetadata.model_validate(
                            content.get("metadata")
                            | {"transient": content.get("transient")}
                        )

    if output:
        cell.outputs.append(output)


def get_default_notebook_path() -> Path:
    """Get a default notebook path in the current working directory."""
    i = 0
    while True:
        notebook_path = Path.cwd() / f"Untitled{f' {i}' if i > 0 else ''}.ipynb"
        if not notebook_path.exists():
            return notebook_path
        i += 1


async def get_default_kernel_id(mkm: AsyncMultiKernelManager) -> str:
    """Get a list of available kernel names."""
    kernel_ids = mkm.list_kernel_ids()
    if len(kernel_ids) > 0:
        return kernel_ids[0]
    return await mkm.start_kernel()


def multiline_string(data: str | list[str]) -> str:
    """Convert data to multiline string."""
    if isinstance(data, str):
        return data
    return "".join(data)


def extract_audio_data_from_html(html: str) -> tuple[str, str] | None:
    """Extract base64 encoded data and mimeType from audio HTML.

    Args:
        html: HTML string containing audio element

    Returns:
        Tuple of (base64_data, mime_type) if found, None otherwise

    Example:
        >>> html = '<audio controls><source src="data:audio/wav;base64,UklGRnoGAABXQVZF..." type="audio/wav"></audio>'
        >>> extract_audio_data_from_html(html)
        ('UklGRnoGAABXQVZF...', 'audio/wav')
    """
    # Pattern to match data URLs in audio elements
    # Matches: data:audio/wav;base64,UklGRnoGAABXQVZF...
    data_url_pattern = r"data:([^;]+);base64,([A-Za-z0-9+/=]+)"

    # Look for data URLs in the HTML
    match = re.search(data_url_pattern, html)
    if match:
        mime_type = match.group(1)
        base64_data = match.group(2)
        return base64_data, mime_type
    return None


def outputs_to_content(path: Path, cell: CodeCell):
    """Convert outputs to string."""
    outputs = cell.outputs
    content: list[ContentBlock] = []
    for output in outputs:
        match output.output_type:
            case "stream":
                content.append(
                    TextContent(
                        type="text",
                        text=output.text
                        if isinstance(output.text, str)
                        else "".join(output.text),
                        _meta={"name": output.name},
                    ),
                )
            case "execute_result" | "display_data":
                for mimetype, data in output.data.items():
                    match mimetype:
                        case "text/plain":
                            content.append(
                                TextContent(
                                    type="text",
                                    text=multiline_string(data),
                                    _meta=output.metadata.model_dump(mode="json"),
                                )
                            )
                        case "text/html":
                            text = multiline_string(data)
                            if (
                                text.strip().startswith("<audio")
                                and text.strip().endswith("</audio>")
                                and (audio_data := extract_audio_data_from_html(text))
                            ):
                                base64_data, mime_type = audio_data
                                content.append(
                                    AudioContent(
                                        type="audio",
                                        data=base64_data,
                                        mimeType=mime_type,
                                        _meta=output.metadata.model_dump(mode="json"),
                                    )
                                )
                            else:
                                content.append(
                                    EmbeddedResource(
                                        type="resource",
                                        resource={
                                            "uri": f"notebook://{path}#{cell.id}",
                                            "mimeType": "text/html",
                                            "text": multiline_string(data),
                                        },
                                    )
                                )
                        case mimetype if mimetype.startswith("image/"):
                            content.append(
                                ImageContent(
                                    type="image",
                                    data=multiline_string(data),
                                    mimeType=mimetype,
                                )
                            )
            case "error":
                raise ToolError(
                    f"Error executing code: {output.ename}: {output.evalue}\n\n".join(
                        output.traceback
                    )
                )
    return content


def create_mcp(mkm: AsyncMultiKernelManager):
    mcp = FastMCP("jupyter")
    mcp.resource(
        "jupyter://kernelspecs", name="kernelspecs", mime_type="application/json"
    )(cast(KernelSpecManager, mkm.kernel_spec_manager).get_all_specs)
    mcp.resource("jupyter://kernels", name="kernels", mime_type="application/json")(
        mkm.list_kernel_ids
    )

    def parse_notebook(path: Path) -> JupyterNotebook:
        return JupyterNotebook.model_validate_json(path.read_text())

    mcp.resource(
        "notebook://{path}", name="notebook", mime_type="application/x-ipynb+json"
    )(parse_notebook)

    mcp.tool(name="restart_kernel")(mkm.restart_kernel)
    mcp.tool(name="shutdown_kernel")(mkm.shutdown_kernel)
    mcp.tool(name="shutdown_all_kernels")(mkm.shutdown_all)
    mcp.tool()(mkm.interrupt_kernel)

    @mcp.tool()
    async def create_notebook(path: Path) -> None:
        """Create a new notebook.

        Args:
            path: Relative path to notebook file
        """
        if path.exists():
            raise FileExistsError(f"Notebook already exists at {path}")
        cwd = Path.cwd()
        if str(cwd) in str(path) or not path.resolve().is_relative_to(Path.cwd()):
            raise ValueError(
                f"Notebook path must be relative to current working directory, got {path}"
            )
        path = path.resolve()

        # Use lock to prevent concurrent creation
        async with _notebook_lock_manager.acquire_lock(path):
            # Double-check file doesn't exist after acquiring lock
            if path.exists():
                raise FileExistsError(f"Notebook already exists at {path}")

            nb = JupyterNotebook.model_validate(
                {
                    "cells": [],
                    "metadata": {
                        "kernelspec": {
                            "display_name": "jupymcp",
                            "language": "python",
                            "name": "python3",
                        },
                        "language_info": {
                            "name": "python",
                            "version": sys.version.split()[0],
                            "mimetype": "text/x-python",
                            "file_extension": ".py",
                            "pygments_lexer": "ipython3",
                            "codemirror_mode": {"name": "ipython", "version": 3},
                        },
                    },
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
            )
            path.write_text(nb.model_dump_json())

    @mcp.tool(structured_output=False)
    async def append_and_execute_cell(
        path: Path,
        code: str,
        silent: bool = False,
        store_history: bool = True,
        user_expressions: dict[str, Any] | None = None,
        stop_on_error: bool = True,
        timeout: float | None = None,
        stdin_hook: str | None = None,
    ) -> str:
        """Append a cell to the notebook and execute it.

        Args:
            path: Path to notebook
            code: Code to execute
            silent: Whether to execute the code silently
            store_history: Whether to store the history
            user_expressions: User expressions to evaluate
            stop_on_error: Whether to stop on error
            timeout: Timeout in seconds
            stdin_hook: Stdin hook

        Returns:
            Content of the cell output
        """
        # Use lock to prevent concurrent modifications to the same notebook
        async with _notebook_lock_manager.acquire_lock(path):
            # Read notebook file
            nb = parse_notebook(path)

            # Create new cell
            cell = CodeCell(
                id=random_cell_id(),
                cell_type="code",
                source=code,
                metadata={},
                outputs=[],
            )
            nb.cells.append(cell)

            # Execute the cell
            km = cast(
                AsyncKernelManager, mkm.get_kernel(await get_default_kernel_id(mkm))
            )
            output_hook = partial(_output_hook, cell)
            reply = await km.client().execute_interactive(
                code=code,
                silent=silent,
                store_history=store_history,
                user_expressions=user_expressions,
                allow_stdin=False,
                stop_on_error=stop_on_error,
                timeout=timeout,
                output_hook=output_hook,
                stdin_hook=stdin_hook,
            )
            cell.execution_count = reply["content"]["execution_count"]

            # Write back to file
            path.write_text(nb.model_dump_json())

            return outputs_to_content(path, cell)

    @mcp.tool()
    async def append_cell(
        path: Path,
        source: str,
        metadata: dict[str, Any] | None = None,
        cell_type: Literal["code", "markdown", "raw"] = "code",
    ) -> int:
        """Append a cell to the notebook.

        Args:
            path: Path to notebook
            source: Cell source
            metadata: Cell metadata
            cell_type: Cell type

        Returns:
            Index of the new cell
        """
        async with _notebook_lock_manager.acquire_lock(path):
            nb = parse_notebook(path)
            cell_id = random_cell_id()
            match cell_type:
                case "code":
                    cell = CodeCell(
                        id=cell_id,
                        cell_type="code",
                        source=source,
                        metadata={},
                        outputs=[],
                    )
                case "markdown":
                    cell = MarkdownCell(
                        id=cell_id,
                        cell_type="markdown",
                        source=source,
                        metadata={},
                    )
                case "raw":
                    cell = RawCell(
                        id=cell_id,
                        cell_type="raw",
                        source=source,
                        metadata={},
                    )
            nb.cells.append(cell)
            path.write_text(nb.model_dump_json())
            return len(nb.cells) - 1

    return mcp


async def main():
    async with kernel_manager() as mkm:
        mcp = create_mcp(mkm)
        await mcp.run_stdio_async()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
