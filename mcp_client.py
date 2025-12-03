# mcp_client.py

import asyncio
import json
from typing import Any, Dict, List, Optional
import os


class MCPClient:
    """MCP Client for communicating with Neo4j MCP Server"""

    def __init__(self, server_path: str, server_args: List[str], neo4j_config: Dict[str, str]):
        self.server_path = server_path
        self.server_args = server_args
        self.neo4j_config = neo4j_config
        self.process: Optional[asyncio.subprocess.Process] = None
        self.tools: List[Dict[str, Any]] = []

    async def start(self):
        """Start the MCP server process"""

        # Start with the current environment
        env = dict(os.environ)

        # Pass Neo4j configuration to the MCP server via environment variables
        # Names expected by mcp-neo4j-cypher
        env.update(
            {
                "NEO4J_URI": self.neo4j_config["uri"],
                "NEO4J_USERNAME": self.neo4j_config["user"],
                "NEO4J_PASSWORD": self.neo4j_config["password"],
                "NEO4J_DATABASE": self.neo4j_config.get("database", "neo4j"),
                "NEO4J_TRANSPORT": "stdio",
            }
        )

        # Debug prints (you can comment these out later)
        print("DEBUG MCP COMMAND:", self.server_path, self.server_args)
        print(
            "DEBUG MCP ENV:",
            {
                "NEO4J_URI": env["NEO4J_URI"],
                "NEO4J_USERNAME": env["NEO4J_USERNAME"],
                "NEO4J_DATABASE": env["NEO4J_DATABASE"],
                "NEO4J_TRANSPORT": env["NEO4J_TRANSPORT"],
            },
        )

        self.process = await asyncio.create_subprocess_exec(
            self.server_path,
            *self.server_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        await self._initialize()

    async def _initialize(self):
        """Initialize MCP connection and retrieve available tools"""
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "neo4j-ollama-agent",
                    "version": "1.0.0",
                },
            },
        }

        await self._send_request(init_request)
        await self._receive_response()

        # List available tools
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }

        await self._send_request(tools_request)
        response = await self._receive_response()

        if response and "result" in response:
            self.tools = response["result"].get("tools", [])

    async def _send_request(self, request: Dict[str, Any]):
        """Send JSON-RPC request to MCP server"""
        if not self.process or not self.process.stdin:
            raise RuntimeError("MCP server not started")

        request_str = json.dumps(request) + "\n"
        self.process.stdin.write(request_str.encode())
        await self.process.stdin.drain()

    async def _receive_response(self) -> Optional[Dict[str, Any]]:
        """Receive JSON-RPC response from MCP server"""
        if not self.process or not self.process.stdout:
            raise RuntimeError("MCP server not started")

        line = await self.process.stdout.readline()
        if line:
            return json.loads(line.decode())
        return None

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call an MCP tool"""

        # Get the current task once and handle None case (for Pylance happiness)
        task = asyncio.current_task()
        task_id = task.get_name() if task is not None else "mcp-call"

        request = {
            "jsonrpc": "2.0",
            "id": task_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

        await self._send_request(request)
        response = await self._receive_response()

        if response and "result" in response:
            return response["result"]
        elif response and "error" in response:
            raise Exception(f"Tool error: {response['error']}")

        return None

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get tools in a generic schema we’ll describe to Ollama"""
        gemini_tools = []

        for tool in self.tools:
            gemini_tool = {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            }

            if "inputSchema" in tool:
                schema = tool["inputSchema"]
                if "properties" in schema:
                    gemini_tool["parameters"]["properties"] = schema["properties"]
                if "required" in schema:
                    gemini_tool["parameters"]["required"] = schema["required"]

            gemini_tools.append(gemini_tool)

        return gemini_tools

    async def stop(self):
        """Stop the MCP server"""
        if self.process:
            self.process.terminate()
            await self.process.wait()
