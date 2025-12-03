# config.py
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration for the Ollama + Neo4j Agent"""

    # Ollama Configuration
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Neo4j Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://7bcd08ff.databases.neo4j.io")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

    # MCP Server Configuration
    MCP_SERVER_PATH = os.getenv("MCP_SERVER_PATH", "mcp-neo4j-cypher")
    MCP_SERVER_ARGS: list[str] = []  # stdio default, no extra args needed

    # Agent Configuration
    MAX_ITERATIONS = 10
    TEMPERATURE = 0.7  # not directly used by Ollama API out of the box but kept for future tuning

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.NEO4J_PASSWORD:
            raise ValueError("NEO4J_PASSWORD is required")

        if not cls.OLLAMA_MODEL:
            raise ValueError("OLLAMA_MODEL is required")
