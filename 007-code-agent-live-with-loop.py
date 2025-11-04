import operator
from typing import TypedDict, Annotated, List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import AnyMessage
from langgraph.graph import StateGraph, END
from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode 

# ====================================================================
# 1. ZUSTAND, TOOLS & LLMS
# ====================================================================

# Zustandsschemata
class SupervisorState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    next: str 
class WorkerState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]

# TOOLS
class CustomCodeInterpreterTool(BaseTool):
    name: str = "execute_code" 
    description: str = "Führt Python-Code aus, um mathematische Probleme oder Datenanalysen zu lösen. Die Eingabe muss der Code-String sein."
    def _run(self, query: str) -> str:
        try:
            output = str(eval(query))
            return f"Code-Ausgabe: {output}"
        except Exception as e:
            return f"FEHLER bei Code-Ausführung: {e}"
    async def _arun(self, query: str) -> str:
        return self._run(query)

tavily_tool = TavilySearch(k=3)
code_tool = CustomCodeInterpreterTool()
web_tools = [tavily_tool]
code_tools_list = [code_tool]

# LLMS
supervisor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
worker_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
supervisor_tools = [code_tool, tavily_tool] 
supervisor_llm_bound = supervisor_llm.bind_tools(supervisor_tools)


# ====================================================================
# 2. WORKER-SUB-GRAPH DEFINITIONEN
# ====================================================================

def _call_llm_worker(state: WorkerState, tools_list: list) -> dict:
    """Standard LLM-Aufruf im Worker-Loop (generiert Tool Call oder Finale Antwort)."""
    llm_with_tools = worker_llm.bind_tools(tools_list)
    response = llm_with_tools.invoke(state["messages"]) 
    return {"messages": [response]}

def _route_worker(state: WorkerState) -> Literal["call_tool", "end"]:
    """Routing für die innere Worker-Schleife."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "call_tool"
    else:
        return "end"


def build_worker_graph(tools_list: list):
    """Erstellt den iterativen Tool-Calling Sub-Graphen."""
    
    workflow = StateGraph(WorkerState)
    tool_node = ToolNode(tools_list)
    
    workflow.add_node("agent", lambda s: _call_llm_worker(s, tools_list))
    workflow.add_node("call_tool", tool_node)
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", _route_worker, {"call_tool": "call_tool", "end": END})
    workflow.add_edge("call_tool", "agent") 
    
    return workflow.compile()

# Die kompilierten Worker-Graphen
web_agent_graph = build_worker_graph(web_tools)
code_agent_graph = build_worker_graph(code_tools_list)

# ====================================================================
# 3. KNOTEN-DEFINITIONEN (HAUPT-SUPERVISOR)
# ====================================================================

# KORREKTUR: Vereinfacht den Worker-Input auf die reine UserMessage

def run_worker_subgraph(graph, state):
    """Führt einen der kompilierten Worker-Graphen aus."""
    print(f"--- WORKER: Delegiert an {graph.get_graph().nodes} ---")
    
    # 🎯 KORREKTUR DER FEHLERQUELLE: Bereinige den Input auf die letzte HumanMessage
    # Nur die initiale Benutzeranfrage wird an den Worker gesendet.
    last_user_message = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
    
    # Der Sub-Graph wird mit der sauberen UserMessage gestartet
    worker_output_state = graph.invoke({"messages": [last_user_message]})
    
    # Das finale Ergebnis des Workers (letzte Nachricht)
    final_message = worker_output_state['messages'][-1]
    
    print("--- WORKER: Aufgabe abgeschlossen ---")
    return {"messages": [final_message], "next": "end"}


# HINWEIS: Diese Funktionen rufen den Sub-Graphen auf
def tavily_search_node(state: SupervisorState) -> dict:
    return run_worker_subgraph(web_agent_graph, state)

def execute_code_node(state: SupervisorState) -> dict:
    return run_worker_subgraph(code_agent_graph, state)


def run_supervisor(state: SupervisorState) -> dict:
    """Der Supervisor-Knoten: Klassifiziert die Anfrage."""
    print("\n--- SUPERVISOR: Klassifiziere Anfrage... ---")
    
    prompt = "Du bist ein Supervisor-Agent. Wähle EINES der verfügbaren Tools (tavily_search oder execute_code), ODER wähle KEIN Tool, wenn die Antwort generisch und einfach ist. Antworte in Deutsch."
    messages = [AIMessage(content=prompt, role="system"), state["messages"][-1]]

    response = supervisor_llm_bound.invoke(messages)
    
    if response.tool_calls:
        chosen_tool = response.tool_calls[0]['name']
        print(f"--- SUPERVISOR-ENTSCHEIDUNG: Delegiere zu '{chosen_tool}' ---")
        return {"messages": [response], "next": chosen_tool}
    else:
        print("--- SUPERVISOR-ENTSCHEIDUNG: Direkte Antwort (END) ---")
        generic_prompt = f"Antworte kurz und direkt auf die Frage: {state['messages'][-1].content}. Antworte in Deutsch."
        final_response = supervisor_llm.invoke([HumanMessage(content=generic_prompt)])
        return {"messages": [final_response], "next": "end"}


def route_supervisor(state: SupervisorState) -> str:
    return state["next"]


# ====================================================================
# 4. GRAPH ERSTELLEN UND KOMPILIEREN (HAUPT-GRAPH)
# ====================================================================

workflow = StateGraph(SupervisorState)

# Knoten hinzufügen (Nutzt die korrekten Node-Namen)
workflow.add_node("supervisor", run_supervisor)
workflow.add_node("tavily_search", tavily_search_node) 
workflow.add_node("execute_code", execute_code_node) 

# Entry Point
workflow.set_entry_point("supervisor")

# Bedingte Kante (Routing)
workflow.add_conditional_edges(
    "supervisor", 
    route_supervisor, 
    {
        "tavily_search": "tavily_search", 
        "execute_code": "execute_code",   
        "end": END                        
    }
)

# Kanten vom Worker zum ENDE
workflow.add_edge("tavily_search", END) 
workflow.add_edge("execute_code", END)

# Kompilieren
app = workflow.compile()

# ====================================================================
# 5. TESTFÄLLE
# ====================================================================

print("\n" + "="*50)
print("MULTI-AGENT SUPERVISOR START - ECHTE ITERATIVE TOOLS")
print("="*50)

# Test 1: Web-Frage (Sollte Tavily wählen)
web_question = "Was sind besten englischsprachigen SciFi Bücher mit Humor im Jahr 2025? Erstelle eine Liste von den Top 5 Büchern mit Titel und Author"
print(f"\n[TEST 1] FRAGE: {web_question}")
final_state_web = app.invoke({"messages": [HumanMessage(content=web_question)]}) 
print(f"FINALE ANTWORT: {final_state_web['messages'][-1].content}")


# Test 2: Code-Frage (Sollte Code wählen)
code_question = "Berechne die Qyadratwurzel aus 81"
print(f"\n[TEST 2] FRAGE: {code_question}")
final_state_code = app.invoke({"messages": [HumanMessage(content=code_question)]})
print(f"FINALE ANTWORT: {final_state_code['messages'][-1].content}")


# Test 3: Allgemeine Frage (Sollte END wählen)
general_question = "Nenne die Hauptfarbe eines Löwenzahns."
print(f"\n[TEST 3] FRAGE: {general_question}")
final_state_general = app.invoke({"messages": [HumanMessage(content=general_question)]})
print(f"FINALE ANTWORT: {final_state_general['messages'][-1].content}")
