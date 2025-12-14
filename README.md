# 🤖 Agentic AI Software Engineer

Coder Buddy is an **Agentic AI-powered software engineer** that generates **fully functional web applications** from natural language prompts.

With a simple prompt like **“Create a calculator web app”** or **“Build a to-do app”**, the system plans, reasons, and generates complete application code — including frontend and core logic — similar to how a human software engineer would work.

---

## 🚀 Features

- Natural language → full web application
- Agentic reasoning and task planning
- Multi-step execution with validation and retries
- Structured, maintainable code generation
- Modular and extensible architecture
- Real-time visibility using AI Agent Debugger

---

## 🧠 How It Works

Coder Buddy uses an **Agentic AI workflow** orchestrated with **LangGraph**:

1. **User Prompt Understanding**  
   Converts the user’s request into a structured plan.

2. **Task Decomposition**  
   Breaks the project into smaller implementation steps.

3. **Code Generation**  
   Generates frontend and application logic using LLM reasoning.

4. **Tool Usage**  
   Uses tools to create files and manage project structure.

5. **Validation & Retry Logic**  
   Automatically retries steps if errors occur.

6. **Final Output**  
   Produces a working web application in the generated project folder.

---

## 🛠 Tech Stack

- **LangGraph** – Agent orchestration & state management  
- **LangChain** – Prompt engineering & tool integration  
- **GPT-OSS** – Reasoning and code generation  
- **Groq** – High-performance inference  
- **Python** – Core implementation

---

## 📁 Project Structure

```text
app-builder/
├── agent/
│   ├── graph.py        # LangGraph workflow
│   ├── prompts.py      # Agent prompts
│   ├── states.py       # Graph states
│   └── tools.py        # Tool definitions (file operations, etc.)
├── generated_project/  # Generated application output
├── main.py             # Entry point
├── README.md
├── pyproject.toml
└── .env
