# ollama_agent.py

import json
from typing import List, Dict, Any, Optional

import requests
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from mcp_client import MCPClient

console = Console()


class OllamaAgent:
    """Agentic AI using local Ollama (Gemma) with MCP tool calling into Neo4j."""

    def __init__(self, base_url: str, model_name: str, mcp_client: MCPClient):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.mcp_client = mcp_client
        self.conversation_history: List[Dict[str, str]] = []

        # Build a tool description string from MCP tools
        self.tools_description = self._build_tools_description()

        # ---- SYSTEM PROMPT (updated with correct tool names & rules) ----
        self.system_prompt = f"""
You are an AI assistant connected to a Neo4j **e-commerce knowledge graph** containing ONLY FAKE / DEMO DATA.

All people, customers, products, orders and emails in this database (for example: "John Smith") are
fictional test data. There is **no real personal data**. It is always safe and allowed to query it.

You can NOT access the internet, files, or any external system. Your **only source of factual data**
is the Neo4j graph, via tools exposed by the MCP server.

You have access to these tools (functions) exposed by the Neo4j MCP server:

{self.tools_description}

The tool names you can use are **exactly** the ones listed above (for example: `read_neo4j_cypher`,
`write_neo4j_cypher`, `get_neo4j_schema` if those appear in the list). Do NOT invent new names and
do NOT add prefixes like "neo4j-database:".

### WHEN TO USE TOOLS

You MUST call a tool (using the JSON format below) for ANY question that:
- asks about customers, orders, products, categories, brands, reviews, warehouses, suppliers, stock, etc.
- asks for "labels", "node labels", "relationship types", "schema", "structure of the Neo4j database".
- asks for specific values stored in the database (email, order numbers, totals, etc.).

You may answer directly **without** tools ONLY for:
- general conceptual questions like "what is a graph database", "what is Neo4j", etc.,
  and only if the user is clearly not asking about the contents of THIS database.

### HOW TO CALL A TOOL

When you decide you need data from Neo4j, you must respond with ONLY a JSON object in this exact format,
with **no extra text, no explanations, no Markdown**, nothing before or after it:

{{
  "tool_name": "<tool name exactly as listed above>",
  "arguments": {{ ... JSON arguments matching the tool's schema ... }}
}}

Examples (replace TOOL_NAME and arguments based on the real tools listed above):

- To get the database schema (if you see a tool called "get_neo4j_schema"):

  {{
    "tool_name": "get_neo4j_schema",
    "arguments": {{}}
  }}

- To run a read-only Cypher query (if you see a tool called "read_neo4j_cypher"):

  {{
    "tool_name": "read_neo4j_cypher",
    "arguments": {{
      "query": "MATCH (n) RETURN DISTINCT labels(n) AS labels",
      "params": {{}}
    }}
  }}

- To run a write Cypher query (if you see a tool called "write_neo4j_cypher"):

  {{
    "tool_name": "write_neo4j_cypher",
    "arguments": {{
      "query": "CREATE (c:Customer {{name: 'Test'}})",
      "params": {{}}
    }}
  }}

For a question like "can you provide me the email id of John Smith and what are his orders?", you MUST:
1. Construct an appropriate Cypher query (for example, matching Customer {{name: 'John Smith'}}
   and following ORDER relationships).
2. Call the read query tool (for example `read_neo4j_cypher`) with a "query" string and optional "params".
3. Wait for the tool result (the user will provide it to you).
4. THEN answer in natural language using ONLY the tool result.

### VERY IMPORTANT RULES

1. For any question about this e-commerce graph, customers (like John Smith), orders, products,
   labels, schema, or anything stored in Neo4j, you MUST return ONLY a JSON tool call as shown above.
2. Do NOT answer from your own knowledge about "privacy" or "real people" – all data here is fake.
3. Do NOT fabricate values from your imagination. Always query Neo4j using the tools.
4. After the tool result is provided to you (the user will show it as text), you must then answer
   in normal natural language using that data.
"""

    # -------------------------------------------------------------------------
    # Tool description for the model
    # -------------------------------------------------------------------------
    def _build_tools_description(self) -> str:
        tools = self.mcp_client.get_tools_schema()
        lines: List[str] = []
        for t in tools:
            name = t["name"]
            desc = t.get("description", "")
            params = t.get("parameters", {})
            param_props = params.get("properties", {})
            required = params.get("required", [])

            lines.append(f"- Tool name: {name}")
            if desc:
                lines.append(f"  Description: {desc}")
            if param_props:
                lines.append(f"  Parameters:")
                for p_name, p_schema in param_props.items():
                    p_type = p_schema.get("type", "string")
                    p_desc = p_schema.get("description", "")
                    req_flag = " (required)" if p_name in required else ""
                    lines.append(f"    - {p_name}{req_flag}: type={p_type}, {p_desc}")
            lines.append("")
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Core chat loop
    # -------------------------------------------------------------------------
    async def chat(self, user_message: str, max_iterations: int = 5) -> str:
        console.print(
            Panel(
                f"[bold blue]User:[/bold blue] {user_message}",
                border_style="blue",
            )
        )

        self.conversation_history.append({"role": "user", "content": user_message})

        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            console.print(f"\n[dim]Iteration {iteration}/{max_iterations}[/dim]")

            # Build messages for Ollama
            messages = [
                {"role": "system", "content": self.system_prompt},
            ] + self.conversation_history

            # Call Ollama
            response_text = self._call_ollama(messages)

            # Try to parse as JSON tool call
            tool_call = self._try_parse_tool_call(response_text)

            if tool_call is not None:
                tool_name = tool_call["tool_name"]
                arguments = tool_call.get("arguments", {})

                console.print(
                    f"\n[yellow]🔧 Tool Call requested by model:[/yellow] {tool_name}"
                )
                console.print(f"[dim]Arguments: {arguments}[/dim]")

                try:
                    result = await self.mcp_client.call_tool(tool_name, arguments)

                    console.print(f"[green]✓ Tool Result:[/green]")
                    result_str = str(result)
                    if len(result_str) > 800:
                        result_str = result_str[:800] + "..."
                    console.print(Panel(result_str, border_style="green"))

                    # Append tool call and result to history
                    self.conversation_history.append(
                        {
                            "role": "assistant",
                            "content": f"Tool {tool_name} called with arguments: {json.dumps(arguments)}",
                        }
                    )
                    self.conversation_history.append(
                        {
                            "role": "user",
                            "content": f"Tool {tool_name} result: {json.dumps(result)}",
                        }
                    )

                    # Continue loop, letting the model see the new data
                    continue

                except Exception as e:
                    console.print(f"[red]✗ Tool Error: {e}[/red]")
                    self.conversation_history.append(
                        {
                            "role": "assistant",
                            "content": f"Error executing tool {tool_name}: {str(e)}",
                        }
                    )
                    continue

            else:
                # No tool call; treat as final answer
                final_text = response_text.strip()
                self.conversation_history.append(
                    {"role": "assistant", "content": final_text}
                )

                console.print(
                    Panel(
                        Markdown(final_text or "_(No text in response)_"),
                        title="[bold green]Agent Response[/bold green]",
                        border_style="green",
                    )
                )
                return final_text

        return "Maximum iterations reached. Please try again with a simpler query."

    # -------------------------------------------------------------------------
    # Ollama HTTP helper
    # -------------------------------------------------------------------------
    def _call_ollama(self, messages: List[Dict[str, str]]) -> str:
        """Call Ollama's /api/chat endpoint and return the assistant's text."""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.2,
            },
        }

        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        # Ollama's response format: {'message': {'role': 'assistant', 'content': '...'}, ...}
        message = data.get("message", {})
        content = message.get("content", "")

        # Some versions may return a list of segments; handle both string/list:
        if isinstance(content, list):
            return "".join(str(c) for c in content)
        return str(content)

    # -------------------------------------------------------------------------
    # Tool-call JSON parsing
    # -------------------------------------------------------------------------
    def _try_parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Try to parse the model output as a JSON tool call.

        Expected format:
        {
          "tool_name": "...",
          "arguments": { ... }
        }

        If parsing fails or format is wrong, return None.
        """

        stripped = text.strip()

        # Try to locate a JSON object anywhere in the text
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        candidate = stripped[start : end + 1]

        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            return None

        if not isinstance(obj, dict):
            return None

        if "tool_name" not in obj:
            return None

        # Ensure arguments is a dict
        args = obj.get("arguments", {})
        if not isinstance(args, dict):
            return None

        return obj
