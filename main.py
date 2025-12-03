# main.py
import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from config import Config
from mcp_client import MCPClient
from ollama_agent import OllamaAgent

console = Console()


async def main():
    """Main application entry point"""

    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
        console.print("[yellow]Please set up your .env file with required values[/yellow]")
        return

    console.print(
        Panel.fit(
            "[bold blue]Neo4j Ollama AI Agent[/bold blue]\n"
            "Agentic AI with MCP Neo4j Integration (Ollama Gemma)",
            border_style="blue",
        )
    )

    # Initialize MCP Client
    console.print("\n[yellow]Initializing MCP Neo4j Server...[/yellow]")
    mcp_client = MCPClient(
        server_path=Config.MCP_SERVER_PATH,
        server_args=Config.MCP_SERVER_ARGS,
        neo4j_config={
            "uri": Config.NEO4J_URI,
            "user": Config.NEO4J_USER,
            "password": Config.NEO4J_PASSWORD,
            "database": Config.NEO4J_DATABASE,
        },
    )

    try:
        await mcp_client.start()
        console.print("[green]✓ MCP Server started successfully[/green]")

        # Display available tools
        tools = mcp_client.get_tools_schema()
        console.print(f"\n[cyan]Available Tools ({len(tools)}):[/cyan]")
        for tool in tools:
            console.print(f"  • {tool['name']}: {tool['description']}")

        # Initialize Ollama Agent
        console.print("\n[yellow]Initializing Ollama Agent...[/yellow]")
        agent = OllamaAgent(
            base_url=Config.OLLAMA_BASE_URL,
            model_name=Config.OLLAMA_MODEL,
            mcp_client=mcp_client,
        )
        console.print("[green]✓ Ollama Agent ready[/green]")

        # Interactive chat loop
        console.print("\n[bold green]Agent is ready! Type 'exit' to quit.[/bold green]\n")

        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")

                if user_input.lower() in ["exit", "quit", "bye"]:
                    console.print("[yellow]Goodbye![/yellow]")
                    break

                if not user_input.strip():
                    continue

                # Process with agent
                await agent.chat(user_input, max_iterations=Config.MAX_ITERATIONS)

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
                continue
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    finally:
        console.print("\n[yellow]Shutting down MCP Server...[/yellow]")
        await mcp_client.stop()
        console.print("[green]✓ Cleanup complete[/green]")


if __name__ == "__main__":
    asyncio.run(main())
