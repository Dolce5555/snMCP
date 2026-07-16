import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
import logging

class MCPClient:
    def __init__(self, logger: logging.Logger, logLevel: int = logging.INFO):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.logger = logger.getChild(self.__class__.__name__) or self.getLogger(f"{__file__.split("/")[-1]}_{self.__class__.__name__}", logLevel)
        self.logLevel = logLevel
        self.tools = None

    def getLogger(self, name: str, logLevel: int = logging.INFO) -> logging.Logger:
        """
        シンプルなロガー初期化ヘルパー。
        アプリ側でハンドラを既に設定している場合は干渉しない。
        """        
        name = f"{__file__.split("/")[-1]}_{self.__class__.__name__}"
        logger = logging.getLogger(name)
        logger.setLevel(logLevel)
        # 既にハンドラが無ければデフォルトの StreamHandler を追加
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logLevel)
            fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
            handler.setFormatter(logging.Formatter(fmt))
            logger.addHandler(handler)
            logger.propagate = False
        return logger

    async def connect(self):
        read_stream, write_stream, _ = await self.exit_stack.enter_async_context(streamable_http_client("http://mcp-server:8000/"))
        self.session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self.session.initialize()
        self.logger.info("MCP server に接続しました")
        self.tools = await self.session.list_tools()
        self.tools = [self.openai_tool(tool) for tool in self.tools.tools]
        self.logger.info(f"使用可能な tools: {[tool['name'] for tool in self.tools]}")

    def openai_tool(self, tool):
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        }


async def main():
    # Connect to a streamable HTTP server
    exit_stack = AsyncExitStack()
    read_stream, write_stream, _ = await exit_stack.enter_async_context(streamable_http_client("http://mcp-server:8000/"))
    session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
    await session.initialize()
    result = await session.call_tool("searchRag", {"query": "watabe"})
    print(result)
    print(result.content[-1].text)
    await exit_stack.aclose()

if __name__ == "__main__":
    asyncio.run(main())