# 📚 Learning LangChain & LangGraph

This repository serves as a personal learning environment and collection of code examples demonstrating the use of the **LangChain** and **LangGraph** frameworks. It documents the journey from basic LLM interactions to the development of complex, stateful agents.

---

## 🚀 Overview of Examples

All scripts are numbered to represent a logical learning path, starting with simple chains and progressing to advanced agent workflows.

| File | Description | Core Concept |
| :--- | :--- | :--- |
| `001-hello-world-joke.py` | A basic "Hello World" example, demonstrating an LLM call and a simple chain. | LLM, PromptTemplate, Chain |
| `002-textdocument-load-and-food-analysis.py` | Loading a document and performing an analysis of its content. | Document Loader, Parsing, Chain |
| `003-langgraph-example.py` | Implementing a basic state graph with LangGraph to make decisions and control the workflow. | LangGraph, State, Graph Node |
| `004-langgraph-tool-agent.py` | Creating an agent that can decide whether to use an external tool based on its state. | Agents, Tools, LangGraph |
| `005-code-agent-simulation.py` | A simulated code agent that creates a plan but only proposes the code execution. | Plan & Execute, Agent Workflow |
| `006-code-agent-live.py` | A functional code agent that generates code and executes it in a live environment. | Live Tool Execution, Agent |
| `007-code-agent-live-with-loop.py` | An advanced version of the code agent using a loop for iterative problem-solving (feedback and correction). | Control Loops, Iterative Agents |

---

## 🛠️ Setup and Installation

This project utilizes **Poetry** for dependency management and **Nix** for a reproducible development environment (`flake.nix`).

### Prerequisites

* Python 3.x
* Git
* (Optional, Recommended) Poetry
* (Optional, Recommended) Nix

### Installation with Poetry

1.  Clone the repository:
    ```bash
    git clone [https://github.com/MacBachi/LearningLangchain.git](https://github.com/MacBachi/LearningLangchain.git)
    cd LearningLangchain
    ```
2.  Install dependencies:
    ```bash
    poetry install
    ```
3.  Activate the virtual environment:
    ```bash
    poetry shell
    ```

### Environment Variables

You will need an API key for the LLM provider you intend to use (e.g., OpenAI). Create a `.env` file in the project root and add your key:

```dotenv
# Example for OpenAI
OPENAI_API_KEY="your-openai-key-here"
