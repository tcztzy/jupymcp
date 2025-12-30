# CLAUDE.md - Project-Specific Instructions for Claude Code

## Project Overview

**JupyMCP** is a Model Context Protocol (MCP) server for Jupyter Notebooks that provides comprehensive notebook interaction through both tools and resources.

- **Language**: Python 3.11+
- **Main Dependencies**: jupyter-client, jupytext, mcp (fastmcp)
- **Architecture**: Async MCP server with multi-kernel support
- **Platform**: Cross-platform (Windows, Linux, macOS)

## Project Structure

```
jupymcp/
├── src/jupymcp/
│   ├── __init__.py
│   ├── __main__.py
│   ├── main.py          # Main MCP server implementation
│   └── model.py         # Generated Pydantic models from nbformat schema
├── scripts/
│   └── generate-model.py # Cross-platform model generation script
├── tests/
│   └── test_main.py
├── gen.sh               # POSIX model generation script (deprecated, use scripts/generate-model.py)
├── pyproject.toml
└── README.md
```

## Key Components

### 1. Main MCP Server (`main.py`)

- **Entry Point**: `main()` function with argparse CLI
- **Transport Modes**: stdio (default), sse, streamable-http
- **Core Manager**: `AsyncMultiKernelManager` for kernel lifecycle management

### 2. Key Features

**Tools**:
- `execute`: Execute code in a kernel with output capture
- `append_cell`: Add cells to notebooks
- `start_kernel`, `restart_kernel`, `shutdown_kernel`: Kernel management
- `create_notebook`: Create new notebook files

**Resources**:
- `jupyter://kernelspecs`: Available kernel specifications
- `jupyter://kernels`: Running kernels list
- `notebook://{path}`: Notebook file access

### 3. Critical Implementation Details

**Concurrency & Locking**:
- `NotebookLockManager`: File-based locks to prevent concurrent notebook modifications
- All notebook write operations MUST use `_notebook_lock_manager.acquire_lock(path)`
- Lock creation is itself protected by `_lock_creation_lock` to prevent races

**Output Handling**:
- `_output_hook`: Captures kernel messages and converts to Pydantic models
- Supports multimedia: text, images (base64), audio (extracted from HTML)
- Error handling: Raises `ToolError` for execution errors

**Cell Management**:
- Uses `nbformat.corpus.words.generate_corpus_id` for cell IDs
- Three cell types: `CodeCell`, `MarkdownCell`, `RawCell`
- Outputs: `ExecuteResult`, `DisplayData`, `Stream`, `Error`

### 4. Generated Models (`model.py`)

- Generated from `nbformat` JSON schema using `datamodel-codegen`
- **DO NOT MANUALLY EDIT** - regenerate using `scripts/generate-model.py`
- Custom class names: `RawCellMetadata`, `MarkdownCellMetadata`, `CodeCellMetadata`, `UnrecognizedCellMetadata`

## Development Guidelines

### Model Generation

When updating models:

```bash
# Cross-platform (recommended)
python scripts/generate-model.py

# POSIX only (deprecated)
bash gen.sh
```

**Process**:
1. Reads `nbformat.v4.schema.json` from virtual environment
2. Runs `datamodel-codegen` with specific Pydantic v2 settings
3. Post-processes to rename `Metadata1-4` classes
4. Removes `Misc` type alias

### Code Style

- Uses Ruff for linting and formatting
- Import sorting enabled (`I` rule)
- Type hints required (Python 3.11+ syntax)
- Async/await patterns throughout

### Testing

```bash
# Run tests with coverage
pytest --cov=src/jupymcp tests/

# Test paths configured in pyproject.toml
# Coverage excludes: pragma: no cover, def main, def amain
```

## Common Development Tasks

### Adding a New Tool

1. Add tool function in `create_mcp()` function
2. Use `@mcp.tool()` decorator
3. Add proper docstring (appears in MCP tool description)
4. Handle async operations appropriately
5. Use locks for notebook modifications
6. Handle errors with `ToolError` for user-facing errors

Example:
```python
@mcp.tool()
async def new_tool(path: Path, param: str) -> str:
    """Tool description shown to LLM clients.

    Args:
        path: Path to notebook
        param: Parameter description

    Returns:
        Result description
    """
    async with _notebook_lock_manager.acquire_lock(path):
        nb = parse_notebook(path)
        # ... implementation
        path.write_text(nb.model_dump_json())
        return "success"
```

### Adding a New Resource

```python
mcp.resource(
    "jupyter://resource_name",
    name="resource_name",
    mime_type="application/json"
)(resource_handler_function)
```

## Important Conventions

### Path Handling

- **Always use `pathlib.Path`** for cross-platform compatibility
- Notebook paths are relative to current working directory
- Validate paths are within CWD for security
- Special path `:memory:` for ephemeral execution (no file write)

### Notebook Operations

1. **Always acquire lock** before reading/writing notebooks
2. Use `parse_notebook(path)` helper for reading
3. Use `model_dump_json(exclude_unset=True)` for writing
4. Create notebooks with proper metadata structure

### Error Handling

- Use `ToolError` for user-facing errors (MCP tool failures)
- Let Python exceptions propagate for system errors
- Validate inputs before acquiring locks
- Provide clear error messages

## Testing Considerations

- Test async operations properly
- Mock `AsyncMultiKernelManager` for kernel tests
- Test concurrent access scenarios (locking)
- Test multimedia output handling
- Test cross-platform path handling

## Platform-Specific Notes

### Windows

- Virtual environment: `.venv/Lib/site-packages`
- Path separator handling via `pathlib`
- sed not available (use Python regex in scripts)

### POSIX (Linux/macOS)

- Virtual environment: `.venv/lib/python*/site-packages`
- gen.sh uses BSD sed syntax (macOS) with `-i ''`

## Dependencies

### Production

- `jupyter-client>=8.7.0`: Kernel client and management
- `jupytext>=1.18.1`: Text-based notebook formats
- `mcp>=1.25.0`: FastMCP framework

### Development

- `datamodel-code-generator`: Model generation from JSON schema
- `nbformat`: Notebook format specs and utilities
- `ipykernel`, `matplotlib`, `pandas`: Testing multimedia outputs
- `pytest-cov`: Test coverage
- `ruff`: Linting and formatting

## Roadmap & Future Work

**Completed**:
- Multi-kernel support
- Multi-notebook support
- Multimedia output support

**Pending**:
- Authentication & security
- Notebook import/export

## Notes for Claude

- This project uses async/await extensively - all MCP handlers should be async
- The locking mechanism is critical - never skip locks for notebook writes
- Model generation is automated - don't manually edit model.py
- Windows support is a first-class requirement - test path handling
- Use existing patterns from main.py when adding new features
- Multimedia handling is complex - refer to `outputs_to_content()` function
- The codebase is relatively small and focused - avoid over-engineering
