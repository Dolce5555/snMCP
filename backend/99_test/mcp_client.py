"""MCP サーバとの stdio 通信テスト用クライアント（LLM なし）"""
import asyncio
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent


def tool_result_text(result: CallToolResult) -> str:
    """CallToolResult からテキスト部分を取り出す"""
    parts: list[str] = []
    for block in result.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        else:
            parts.append(str(block))
    return "\n".join(parts)


class MCPClient:
    def __init__(self, server_script: str | Path):
        self.server_script = Path(server_script)
        self.exit_stack = AsyncExitStack()
        self.session: ClientSession | None = None

    async def connect(self) -> None:
        """stdio で MCP サーバを起動し、セッションを初期化"""
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(self.server_script)],
            cwd=str(self.server_script.parent),
        )
        read_stream, write_stream = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self.session.initialize()

    async def list_tools(self) -> None:
        """登録ツール名の一覧を返す"""
        if self.session is None:
            raise RuntimeError("connect() を先に呼んでください")
        response = await self.session.list_tools()
        for tool in response.tools:
            print(tool.name, tool.description)

    async def call_tool(self, name: str, arguments: dict | None = None) -> CallToolResult:
        """ツールを直接実行"""
        if self.session is None:
            raise RuntimeError("connect() を先に呼んでください")
        return await self.session.call_tool(name, arguments or {})

    async def close(self) -> None:
        await self.exit_stack.aclose()
        self.session = None

async def main():
    client = MCPClient("/app/backend/99_test/mcp_server.py")
    try:
        await client.connect()
        result = await client.list_tools()
        print(result)
        result = await client.call_tool("test_communication", {"message": "私の名前はSN"})
        print(result)
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())