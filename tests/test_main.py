"""Tests for jupymcp.main module."""

import asyncio
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jupymcp.main import (
    NotebookLockManager,
    _output_hook,
    extract_audio_data_from_html,
    get_default_kernel_id,
    get_default_notebook_path,
    kernel_manager,
    multiline_string,
    outputs_to_content,
)


class TestNotebookLockManager:
    """Test NotebookLockManager class."""

    def test_init(self):
        """Test initialization."""
        manager = NotebookLockManager()
        assert manager._locks == {}
        assert isinstance(manager._lock_creation_lock, asyncio.Lock)

    async def test_get_lock_new(self):
        """Test getting a new lock."""
        manager = NotebookLockManager()
        notebook_path = Path("/tmp/test.ipynb")
        lock = await manager.get_lock(notebook_path)
        assert isinstance(lock, asyncio.Lock)
        assert str(notebook_path.resolve()) in manager._locks

    async def test_get_lock_existing(self):
        """Test getting an existing lock."""
        manager = NotebookLockManager()
        notebook_path = Path("/tmp/test.ipynb")
        lock1 = await manager.get_lock(notebook_path)
        lock2 = await manager.get_lock(notebook_path)
        assert lock1 is lock2

    async def test_acquire_lock(self):
        """Test acquire_lock context manager."""
        manager = NotebookLockManager()
        notebook_path = Path("/tmp/test.ipynb")

        async with manager.acquire_lock(notebook_path) as acquired:
            # Should not raise exception
            assert acquired is None
            # Lock should be held
            lock = await manager.get_lock(notebook_path)
            assert lock.locked()


class TestUtilityFunctions:
    """Test utility functions."""

    def test_multiline_string_string(self):
        """Test multiline_string with string input."""
        result = multiline_string("hello")
        assert result == "hello"

    def test_multiline_string_list(self):
        """Test multiline_string with list input."""
        result = multiline_string(["line1", "line2", "line3"])
        assert result == "line1line2line3"

    def test_extract_audio_data_from_html_found(self):
        """Test extract_audio_data_from_html with valid audio HTML."""
        html = '<audio controls><source src="data:audio/wav;base64,UklGRnoGAABXQVZF..." type="audio/wav"></audio>'
        result = extract_audio_data_from_html(html)
        assert result == ("UklGRnoGAABXQVZF", "audio/wav")

    def test_extract_audio_data_from_html_not_found(self):
        """Test extract_audio_data_from_html without audio data."""
        html = "<div>No audio here</div>"
        result = extract_audio_data_from_html(html)
        assert result is None

    def test_get_default_notebook_path(self, tmp_path):
        """Test get_default_notebook_path."""
        original_cwd = Path.cwd()
        try:
            # Change to temporary directory
            os.chdir(tmp_path)

            # First call should return "Untitled.ipynb"
            path1 = get_default_notebook_path()
            assert path1.name == "Untitled.ipynb"

            # Create that file
            path1.touch()

            # Next call should return "Untitled 1.ipynb"
            path2 = get_default_notebook_path()
            assert path2.name == "Untitled 1.ipynb"

            # Create that file
            path2.touch()

            # Next call should return "Untitled 2.ipynb"
            path3 = get_default_notebook_path()
            assert path3.name == "Untitled 2.ipynb"
        finally:
            os.chdir(original_cwd)


class TestOutputsToContent:
    """Test outputs_to_content function."""

    def test_outputs_to_content_stream(self):
        """Test converting Stream output."""
        from jupymcp.model import Stream

        path = Path("/tmp/test.ipynb")
        cell = MagicMock()
        cell.outputs = [
            Stream(output_type="stream", name="stdout", text="Hello, world!")
        ]

        result = outputs_to_content(path, cell)
        assert len(result) == 1
        assert result[0].type == "text"
        assert result[0].text == "Hello, world!"
        # meta may be None or contain name
        if hasattr(result[0], "meta") and result[0].meta is not None:
            assert result[0].meta.get("name") == "stdout"

    def test_outputs_to_content_text_plain(self):
        """Test converting ExecuteResult with text/plain."""
        from jupymcp.model import ExecuteResult

        path = Path("/tmp/test.ipynb")
        cell = MagicMock()
        cell.outputs = [
            ExecuteResult(
                output_type="execute_result",
                execution_count=1,
                data={"text/plain": "Result text"},
                metadata={},
            )
        ]

        result = outputs_to_content(path, cell)
        assert len(result) == 1
        assert result[0].type == "text"
        assert result[0].text == "Result text"

    def test_outputs_to_content_error_raises(self):
        """Test that Error output raises ToolError."""
        from mcp.server.fastmcp.exceptions import ToolError

        from jupymcp.model import Error

        path = Path("/tmp/test.ipynb")
        cell = MagicMock()
        cell.outputs = [
            Error(
                output_type="error",
                ename="ValueError",
                evalue="Invalid value",
                traceback=["Traceback line 1", "Traceback line 2"],
            )
        ]

        with pytest.raises(ToolError):
            outputs_to_content(path, cell)

    def test_outputs_to_content_text_html(self):
        """Test converting ExecuteResult with text/html."""
        from jupymcp.model import ExecuteResult

        path = Path("/tmp/test.ipynb")
        cell = MagicMock()
        cell.id = "test_cell"
        cell.outputs = [
            ExecuteResult(
                output_type="execute_result",
                execution_count=1,
                data={"text/html": "<div>Hello</div>"},
                metadata={},
            )
        ]

        result = outputs_to_content(path, cell)
        assert len(result) == 1
        assert result[0].type == "resource"
        assert result[0].resource.mimeType == "text/html"
        assert result[0].resource.text == "<div>Hello</div>"
        assert str(path) in str(result[0].resource.uri)

    def test_outputs_to_content_text_html_audio(self):
        """Test converting ExecuteResult with audio HTML."""
        from jupymcp.model import ExecuteResult

        path = Path("/tmp/test.ipynb")
        cell = MagicMock()
        cell.id = "test_cell"
        cell.outputs = [
            ExecuteResult(
                output_type="execute_result",
                execution_count=1,
                data={
                    "text/html": '<audio controls><source src="data:audio/wav;base64,ABCD" type="audio/wav"></audio>'
                },
                metadata={},
            )
        ]

        result = outputs_to_content(path, cell)
        assert len(result) == 1
        assert result[0].type == "audio"
        assert result[0].mimeType == "audio/wav"
        assert result[0].data == "ABCD"

    def test_outputs_to_content_image(self):
        """Test converting ExecuteResult with image data."""
        from jupymcp.model import ExecuteResult

        path = Path("/tmp/test.ipynb")
        cell = MagicMock()
        cell.outputs = [
            ExecuteResult(
                output_type="execute_result",
                execution_count=1,
                data={"image/png": "base64_image_data"},
                metadata={},
            )
        ]

        result = outputs_to_content(path, cell)
        assert len(result) == 1
        assert result[0].type == "image"
        assert result[0].mimeType == "image/png"
        assert result[0].data == "base64_image_data"

    def test_outputs_to_content_display_data(self):
        """Test converting DisplayData output."""
        from jupymcp.model import DisplayData

        path = Path("/tmp/test.ipynb")
        cell = MagicMock()
        cell.outputs = [
            DisplayData(
                output_type="display_data",
                data={"text/plain": "display text"},
                metadata={},
            )
        ]

        result = outputs_to_content(path, cell)
        assert len(result) == 1
        assert result[0].type == "text"
        assert result[0].text == "display text"


class TestOutputHook:
    """Test _output_hook function."""

    def test_output_hook_execute_result(self):
        """Test _output_hook with execute_result message."""
        from jupymcp.model import CodeCell, ExecuteResult

        cell = CodeCell(
            id="test",
            cell_type="code",
            source="test",
            metadata={},
            outputs=[],
            execution_count=None,
        )

        message = {
            "header": {"msg_type": "execute_result"},
            "content": {
                "metadata": {"test": "metadata"},
                "data": {"text/plain": "result"},
                "execution_count": 1,
            },
        }

        _output_hook(cell, message)

        assert len(cell.outputs) == 1
        output = cell.outputs[0]
        assert isinstance(output, ExecuteResult)
        assert output.output_type == "execute_result"
        assert output.execution_count == 1
        assert output.data == {"text/plain": "result"}

    def test_output_hook_stream(self):
        """Test _output_hook with stream message."""
        from jupymcp.model import CodeCell, Stream

        cell = CodeCell(
            id="test",
            cell_type="code",
            source="test",
            metadata={},
            outputs=[],
            execution_count=None,
        )

        message = {
            "header": {"msg_type": "stream"},
            "content": {"name": "stdout", "text": "Hello world"},
        }

        _output_hook(cell, message)

        assert len(cell.outputs) == 1
        output = cell.outputs[0]
        assert isinstance(output, Stream)
        assert output.output_type == "stream"
        assert output.name == "stdout"
        assert output.text == "Hello world"

    def test_output_hook_display_data(self):
        """Test _output_hook with display_data message."""
        from jupymcp.model import CodeCell, DisplayData

        cell = CodeCell(
            id="test",
            cell_type="code",
            source="test",
            metadata={},
            outputs=[],
            execution_count=None,
        )

        message = {
            "header": {"msg_type": "display_data"},
            "content": {
                "metadata": {"test": "meta"},
                "data": {"text/plain": "display"},
                "transient": {"display_id": "123"},
            },
        }

        _output_hook(cell, message)

        assert len(cell.outputs) == 1
        output = cell.outputs[0]
        assert isinstance(output, DisplayData)
        assert output.output_type == "display_data"
        assert output.data == {"text/plain": "display"}

    def test_output_hook_error(self):
        """Test _output_hook with error message."""
        from jupymcp.model import CodeCell, Error

        cell = CodeCell(
            id="test",
            cell_type="code",
            source="test",
            metadata={},
            outputs=[],
            execution_count=None,
        )

        message = {
            "header": {"msg_type": "error"},
            "content": {
                "ename": "ValueError",
                "evalue": "Invalid value",
                "traceback": ["line1", "line2"],
            },
        }

        _output_hook(cell, message)

        assert len(cell.outputs) == 1
        output = cell.outputs[0]
        assert isinstance(output, Error)
        assert output.output_type == "error"
        assert output.ename == "ValueError"
        assert output.evalue == "Invalid value"
        assert output.traceback == ["line1", "line2"]

    def test_output_hook_clear_output(self):
        """Test _output_hook with clear_output message."""
        from jupymcp.model import CodeCell, Stream

        cell = CodeCell(
            id="test",
            cell_type="code",
            source="test",
            metadata={},
            outputs=[Stream(output_type="stream", name="stdout", text="test")],
            execution_count=None,
        )

        assert len(cell.outputs) == 1

        message = {"header": {"msg_type": "clear_output"}, "content": {}}

        _output_hook(cell, message)

        assert len(cell.outputs) == 0

    def test_output_hook_update_display_data(self):
        """Test _output_hook with update_display_data message."""
        from jupymcp.model import CodeCell, DisplayData, OutputMetadata

        cell = CodeCell(
            id="test",
            cell_type="code",
            source="test",
            metadata={},
            outputs=[
                DisplayData(
                    output_type="display_data",
                    data={"text/plain": "old"},
                    metadata=OutputMetadata.model_validate(
                        {"transient": {"display_id": "123"}}
                    ),
                )
            ],
            execution_count=None,
        )

        message = {
            "header": {"msg_type": "update_display_data"},
            "content": {
                "data": {"text/plain": "new"},
                "metadata": {"updated": True},
                "transient": {"display_id": "123"},
            },
        }

        _output_hook(cell, message)

        assert len(cell.outputs) == 1
        output = cell.outputs[0]
        assert output.data == {"text/plain": "new"}


class TestGetDefaultKernelId:
    """Test get_default_kernel_id function."""

    async def test_get_default_kernel_id_existing(self):
        """Test get_default_kernel_id when kernels exist."""
        from unittest.mock import MagicMock

        mock_mkm = MagicMock()
        mock_mkm.list_kernel_ids.return_value = ["kernel1", "kernel2"]
        mock_mkm.start_kernel = MagicMock()

        result = await get_default_kernel_id(mock_mkm)

        assert result == "kernel1"
        mock_mkm.list_kernel_ids.assert_called_once()
        mock_mkm.start_kernel.assert_not_called()

    async def test_get_default_kernel_id_start_new(self):
        """Test get_default_kernel_id when no kernels exist."""
        from unittest.mock import AsyncMock, MagicMock

        mock_mkm = MagicMock()
        mock_mkm.list_kernel_ids.return_value = []
        mock_mkm.start_kernel = AsyncMock(return_value="new_kernel")

        result = await get_default_kernel_id(mock_mkm)

        assert result == "new_kernel"
        mock_mkm.list_kernel_ids.assert_called_once()
        mock_mkm.start_kernel.assert_called_once()


class TestKernelManager:
    """Test kernel_manager context manager."""

    async def test_kernel_manager(self):
        """Test kernel_manager context manager."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_km = AsyncMock()
        mock_km.shutdown_all = AsyncMock()

        with (
            patch(
                "jupymcp.main.AsyncMultiKernelManager", return_value=mock_km
            ) as mock_mkm_class,
            patch(
                "jupymcp.main.KernelSpecManager", return_value=MagicMock()
            ) as mock_ksm_class,
        ):
            async with kernel_manager() as km:
                assert km is mock_km
                mock_mkm_class.assert_called_once()
                mock_ksm_class.assert_called_once()
                assert mock_km.kernel_spec_manager is not None

            # Ensure shutdown_all was called
            mock_km.shutdown_all.assert_called_once_with(now=True)
