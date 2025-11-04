import operator
from typing import TypedDict, Annotated, List

# Core LangChain Komponenten
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults

# LangGraph Komponenten
from langgraph.graph.message import AnyMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode # <-- NEU: Stabile Komponente für Tool-Ausführung

# ====================================================================
# 1. SETUP: ZUSTAND, TOOLS UND LLM
# ====================================================================

# Zustandsschema: Speichert nur Nachrichten
class AgentState(TypedDict):
    # Operator.add stellt sicher, dass neue Nachrichten angehängt werden (Gedächtnis)
    messages: Annotated[List[AnyMessage], operator.add]

# Tool-Definition: Tavily Websuche
search_tool = TavilySearchResults(max_results=3)
tools = [search_tool]

# LLM initialisieren und mit Tools BINDEN
# Die .bind_tools()-Methode macht das LLM zu einem "Tool-Calling"-Modell.
llm_with_tools = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

# ====================================================================
# 2. KNOTEN-DEFINITIONEN
# ====================================================================

# Knoten 1: LLM-Aufruf (Entscheider und Antwortgenerator)
def call_llm_or_tool(state: AgentState) -> dict:
    """Führt den LLM-Aufruf durch und gibt entweder eine Antwort oder einen Tool-Call zurück."""
    print("\n--- NODE: LLM entscheidet über Aktion (Antwort/Tool) ---")
    
    # Sende den Chat-Verlauf an das LLM
    result = llm_with_tools.invoke(state["messages"])
    
    # Fügt die AI-Antwort oder den Tool-Call-Vorschlag zum Zustand hinzu
    return {"messages": [result]}

# Routing-Funktion: Kontrolliert den Fluss
def route_next(state: AgentState) -> str:
    """Entscheidet, ob der Graph zum Tool geht oder endet."""
    last_message = state["messages"][-1]
    
    if last_message.tool_calls:
        # Das LLM hat einen Tool-Call vorgeschlagen: Gehe zum Tool-Knoten
        print("--- ROUTER: Tool-Call gefunden. Gehe zu Tool-Node.")
        return "call_tool_node"
    else:
        # Das LLM hat eine finale AIMessage generiert: Beende den Graphen
        print("--- ROUTER: Keine Tool-Calls. ENDE.")
        return "end"

# ====================================================================
# 3. GRAPH ERSTELLEN UND KOMPILIEREN
# ====================================================================

workflow = StateGraph(AgentState)

# Knoten hinzufügen
workflow.add_node("call_llm_node", call_llm_or_tool)

# Der ToolNode kapselt die gesamte Logik von ToolExecutor
workflow.add_node("call_tool_node", ToolNode(tools)) 

# 1. Entry Point
workflow.set_entry_point("call_llm_node")

# 2. Bedingte Kante (Nach LLM-Aufruf entscheidet der Router)
workflow.add_conditional_edges(
    "call_llm_node",    # QUELLE
    route_next,         # REGEL: Entscheide, ob Tool oder Ende
    {
        "call_tool_node": "call_tool_node", # Wenn Tool nötig, gehe zum ToolNode
        "end": END                          # Wenn fertig, beende den Graphen
    }
)

# 3. Normale Kante (Nach Tool-Ausführung IMMER zurück zum LLM für finale Antwort)
# Der Loop: Das LLM muss das Tool-Ergebnis sehen, um die finale Antwort zu generieren.
workflow.add_edge("call_tool_node", "call_llm_node")

# Kompilieren
app = workflow.compile()

# ====================================================================
# 4. AUSFÜHRUNG
# ====================================================================

# Frage, die eine Websuche erfordert
question = "Was ist das aktuelle meistverkaufte Buch in Deutschland im Bereich Science Fiction?"

print(f"\n--- STARTE AGENT MIT FRAGE: {question} ---")

# Aufruf des Agenten
final_state = app.invoke({"messages": [HumanMessage(content=question)]})

# Ausgabe der finalen Antwort
final_response = final_state['messages'][-1].content
print("\n--- FINALE ANTWORT DES AGENTEN ---")
print(final_response)
