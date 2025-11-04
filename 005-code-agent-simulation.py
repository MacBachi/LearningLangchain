import operator
from typing import TypedDict, Annotated, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import AnyMessage
from langgraph.graph import StateGraph, END
from langchain_core.tools import BaseTool
##from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
# ====================================================================
# 1. SETUP & ZUSTAND
# ====================================================================

# Zustandsschema: Speichert Nachrichten und die Supervisor-Entscheidung
class SupervisorState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    next: str 

# KORREKTUR: Manuelles Code-Tool 
class CustomCodeInterpreterTool(BaseTool):
    name: str = "execute_code" # Wird oft ignoriert
    description: str = "Führt Python-Code aus, um mathematische Probleme oder Datenanalysen zu lösen. Muss für Berechnungen verwendet werden."
    
    def _run(self, query: str) -> str:
        return f"Code-Agent hat die Berechnung für '{query}' simuliert."
    async def _arun(self, query: str) -> str:
        return self._run(query)

tavily_tool = TavilySearch(k=3)
##tavily_tool = TavilySearchResults(max_results=3)
supervisor_tools = [CustomCodeInterpreterTool, tavily_tool] 
supervisor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
supervisor_llm_bound = supervisor_llm.bind_tools(supervisor_tools)

# ====================================================================
# 2. KNOTEN-DEFINITIONEN (AKTIONEN)
# ====================================================================

def run_supervisor(state: SupervisorState) -> dict:
    """Der Supervisor-Knoten: Klassifiziert die Anfrage und setzt 'next'."""
    print("\n--- SUPERVISOR: Klassifiziere Anfrage... ---")
    
    prompt = (
        "Du bist ein Supervisor-Agent. Wähle EINES der verfügbaren Tools (tavily_search oder execute_code), "
        "ODER wähle KEIN Tool, wenn die Antwort generisch und einfach ist."
    )
    
    messages = [
        AIMessage(content=prompt, role="system"),
        state["messages"][-1]
    ]

    response = supervisor_llm_bound.invoke(messages)
    
    if response.tool_calls:
        # Hier wird der Name extrahiert. Das LLM liefert im Fehlerfall 'CustomCodeInterpreterTool'
        chosen_tool = response.tool_calls[0]['name']
        print(f"--- SUPERVISOR-ENTSCHEIDUNG: Delegiere zu '{chosen_tool}' ---")
        return {"messages": [response], "next": chosen_tool}
    else:
        # Direkte Antwort für generische Fragen
        print("--- SUPERVISOR-ENTSCHEIDUNG: Direkte Antwort (END) ---")
        generic_prompt = f"Antworte kurz und direkt auf die Frage: {state['messages'][-1].content}"
        final_response = supervisor_llm.invoke([HumanMessage(content=generic_prompt)])
        
        return {"messages": [final_response], "next": "end"}


def web_search_agent(state: SupervisorState) -> dict:
    """Simulierter Knoten des Web-Recherche Agenten."""
    print("--- WORKER: Web-Recherche Agent aktiv ---")
    final_answer = "Der Web-Recherche Agent hat die neuesten Daten gefunden und liefert die fundierte Antwort." 
    return {"messages": [AIMessage(content=final_answer)], "next": "end"}


def execute_code_agent(state: SupervisorState) -> dict:
    """Simulierter Knoten des Code-Agenten."""
    print("--- WORKER: Code-Agent aktiv ---")
    final_answer = "Der Code-Agent hat die Berechnung erfolgreich durchgeführt und liefert das Ergebnis." 
    return {"messages": [AIMessage(content=final_answer)], "next": "end"}


def route_supervisor(state: SupervisorState) -> str:
    return state["next"]

# ====================================================================
# 4. GRAPH ERSTELLEN UND KOMPILIEREN
# ====================================================================

workflow = StateGraph(SupervisorState)

# 1. Knoten hinzufügen
# KORREKTUR: Wir registrieren den Knoten mit dem Namen, den das LLM im Fehlerfall gewählt hat:
workflow.add_node("CustomCodeInterpreterTool", execute_code_agent) # <--- Korrigiert
workflow.add_node("tavily_search", web_search_agent)
workflow.add_node("supervisor", run_supervisor) 

# 2. Entry Point
workflow.set_entry_point("supervisor")

# 3. Bedingte Kante (Routing)
workflow.add_conditional_edges(
    "supervisor", 
    route_supervisor, 
    {
        "tavily_search": "tavily_search",
        "CustomCodeInterpreterTool": "CustomCodeInterpreterTool", # <--- Korrigierter Schlüssel
        "end": END                        
    }
)

# 4. Kanten vom Worker zum ENDE
workflow.add_edge("tavily_search", END) 
workflow.add_edge("CustomCodeInterpreterTool", END) # <--- Korrigiert

# Kompilieren
app = workflow.compile()

# ====================================================================
# 5. TESTFÄLLE
# ====================================================================

print("\n" + "="*50)
print("MULTI-AGENT SUPERVISOR START")
print("="*50)

# Test 1: Web-Frage (Sollte Tavily wählen)
web_question = "Wer hat die letzte Fußball-Weltmeisterschaft gewonnen? Ich brauche aktuelle Infos."
print(f"\n[TEST 1] FRAGE: {web_question}")
final_state_web = app.invoke({"messages": [HumanMessage(content=web_question)]}) 
print(f"FINALE ANTWORT: {final_state_web['messages'][-1].content}")


# Test 2: Code-Frage (Sollte Code wählen)
code_question = "Was ist 123 multipliziert mit 456?"
print(f"\n[TEST 2] FRAGE: {code_question}")
final_state_code = app.invoke({"messages": [HumanMessage(content=code_question)]})
print(f"FINALE ANTWORT: {final_state_code['messages'][-1].content}")


# Test 3: Allgemeine Frage (Sollte END wählen und direkt antworten)
general_question = "Was ist die Hauptstadt von Deutschland?"
print(f"\n[TEST 3] FRAGE: {general_question}")
final_state_general = app.invoke({"messages": [HumanMessage(content=general_question)]})
print(f"FINALE ANTWORT: {final_state_general['messages'][-1].content}")


