# Neo4j Ollama AI Agent

An **agentic AI application** that interacts with a **Neo4j E-Commerce Knowledge Graph** using **Model Context Protocol (MCP)** and runs entirely on your local machine with **Ollama’s Gemma 3:1B model**.

---

## 🚀 Stack & Integrations

- **LLM:** Ollama (Gemma 3:1B)
- **Graph DB:** Neo4j (E-Commerce Knowledge Graph)
- **Tool Protocol:** Model Context Protocol (MCP)
- **Design:** Agentic Loop + Function/Tool Calling
- **Terminal UI:** Rich, Panels, Markdown Rendering
- **Transport:** STDIO (MCP ↔ Neo4j), HTTP (Agent ↔ Ollama)

---

## 🎯 What This Agent Can Do

The agent can **understand your query intent** and dynamically call Neo4j tools to:

- Inspect the graph schema & node labels
- Generate and execute **read-only and write Cypher queries**
- Fetch customer order history, product insights, supply chain, reviews, stock, affinity patterns, and recommendations
- Return grounded, structured database responses instead of hallucinated data

---

## 📁 Project Structure

neo4j-ollama-agent/
├── main.py
├── config.py
├── mcp_client.py
├── ollama_agent.py
├── requirements.txt
├── .env
└── README.md

---

## 🛠 Installation Guide (Windows)

### 1. Clone or create your project folder

```bat
E:
mkdir neo4j-ollama-agent
cd neo4j-ollama-agent

### 2. Create and activate Python venv

python -m venv venv
venv\Scripts\activate

### 3. Install Python dependencies

pip install --upgrade pip
pip install -r requirements.txt

### 4. Install Neo4j MCP Cypher server

pip install mcp-neo4j-cypher

### 5. Configure environment

OLLAMA_MODEL= <your_model>
OLLAMA_BASE_URL=http://localhost:11434

NEO4J_URI=<your_uri>
NEO4J_USER=<your_username>
NEO4J_PASSWORD=<your_password>
NEO4J_DATABASE=<your_db_name>

MCP_SERVER_PATH=mcp-neo4j-cypher

### 6. Make sure Neo4j & Ollama are running

Neo4j:

Open Neo4j Desktop or service and start your database

Ollama (separate terminal):

ollama pull <your_model_name>
ollama serve

### 7. Start the agent

python main.py

## 💬 Example Usage

You: What are the node labels in this graph?
You: Give me the email ID of John Smith and list his orders
You: Show top-rated products by brand
You: What products are frequently bought together with MacBook Pro 16"?

## ⚠ Notes & Best Practices

This project contains only DEMO/FAKE data, safe to query

Never commit your .env file containing credentials
Add it to .gitignore:

echo .env >> .gitignore

To reset linter issues in VS Code, reload window → Ctrl+Shift+P → Developer: Reload Window

