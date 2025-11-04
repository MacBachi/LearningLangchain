import operator
from typing import TypedDict, Annotated, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import AnyMessage
from langgraph.graph import StateGraph, END
from langchain_core.tools import BaseTool
from langchain_tavily import TavilySearch # Korrigierter Import (löst Deprecation Warning)
from langgraph.prebuilt import ToolNode # Benötigt für die reale Tool-Ausführung
from langchain_core.runnables import RunnablePassthrough # Benötigt für die finale Antwortgenerierung

# ====================================================================
# 1. SETUP & ZUSTAND
# ====================================================================

# Zustandsschema: Speichert Nachrichten und die Supervisor-Entscheidung
class SupervisorState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    next: str 

# KORREKTUR: Manuelles Code-Tool (führt Code aus, Pydantic V2-konform)
class CustomCodeInterpreterTool(BaseTool):
    name: str = "execute_code" 
    description: str = "Führt Python-Code aus, um mathematische Probleme oder Datenanalysen zu lösen. Die Eingabe muss der Code-String sein."
    
    def _run(self, query: str) -> str:
        try:
            # ECHTE CODE-AUSFÜHRUNG: eval() des vom LLM generierten Codes
            output = str(eval(query))
            return f"Code-Ausgabe: {output}"
        except Exception as e:
            return f"FEHLER bei Code-Ausführung: {e}"
    async def _arun(self, query: str) -> str:
        return self._run(query)

# Die Tools für die Worker-Agenten
tavily_tool = TavilySearch(k=3) # Korrigierte Initialisierung
code_tool = CustomCodeInterpreterTool()
web_tools = [tavily_tool]
code_tools_list = [code_tool]

# LLMs (Supervisor + Worker)
supervisor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
worker_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Binde die "Agenten-Tools" an das Supervisor LLM zur KLASSIFIKATION
supervisor_tools = [code_tool, tavily_tool] 
supervisor_llm_bound = supervisor_llm.bind_tools(supervisor_tools)

# ====================================================================
# 2. KNOTEN-DEFINITIONEN (ECHTE AKTIONEN)
# ====================================================================

# Worker: Der Web-Recherche Agent (ECHTE WEBSUCHE)
def web_search_agent(state: SupervisorState) -> dict:
    """Führt die Websuche aus, erhält die Ergebnisse und generiert die finale Antwort."""
    print("--- WORKER: Web-Recherche Agent: Startet ECHTE Suche ---")
    
    # Der Worker LLM (muss die Websuche durchführen können)
    llm_with_search = worker_llm.bind_tools(web_tools)
    tool_node = ToolNode(web_tools)

    # 1. LLM entscheidet, wie es suchen soll (generiert Tool Call)
    search_prompt = HumanMessage(content=f"Führe eine Websuche durch für die ursprüngliche Frage: {state['messages'][0].content}")
    search_call_response = llm_with_search.invoke([search_prompt])

    # 2. Tool wird ausgeführt und Ergebnis generiert
    tool_output = tool_node.invoke({"messages": [search_call_response]})

    # 3. Finales LLM fasst zusammen
    final_prompt_messages = [
        HumanMessage(content=f"Generiere eine finale, fundierte Antwort basierend NUR auf den folgenden Suchergebnissen. Ursprüngliche Frage: {state['messages'][0].content}")
    ] + [search_call_response] + tool_output['messages']
    
    final_response = worker_llm.invoke(final_prompt_messages)

    print("--- WORKER: Web-Recherche abgeschlossen ---")
    return {"messages": [final_response], "next": "end"}


# Worker: Der Code-Agent (ECHTE CODE-AUSFÜHRUNG)
def execute_code_agent(state: SupervisorState) -> dict:
    """Führt Code aus, erhält das Ergebnis und generiert die finale Antwort."""
    print("--- WORKER: Code-Agent: Startet ECHTE Code-Ausführung ---")
    
    # 1. LLM generiert den Code-String
    code_generation_prompt = HumanMessage(content=f"Schreibe NUR den Python-Code (keine Erklärungen, kein Markdown), um dies zu berechnen: {state['messages'][-1].content}")
    code_string = worker_llm.invoke([code_generation_prompt]).content.strip()

    # 2. Das Code-Tool wird ausgeführt
    code_tool_instance = CustomCodeInterpreterTool()
    code_output = code_tool_instance.invoke(code_string)

    # 3. Finales LLM fasst zusammen
    final_prompt_messages = [
        # 1. Die ursprüngliche User-Nachricht (optional, aber hilfreich für Kontext)
        state['messages'][0], 
        # 2. Die Anweisung, die finale Antwort zu generieren
        HumanMessage(content=f"Generiere eine finale, klare Antwort. Ursprüngliche Frage: {state['messages'][-1].content}"), 
        # 3. Der Code-Output als einfache AIMessage (oder HumanMessage), um Tool-Rollen zu umgehen
        AIMessage(content=f"Code-Ausgabe zur Lösung der Aufgabe: {code_output}"),
    ]   
    
    final_response = worker_llm.invoke(final_prompt_messages)

    print("--- WORKER: Code-Ausführung abgeschlossen ---")
    return {"messages": [final_response], "next": "end"}


# Supervisor: Der Entscheider-Knoten (Unverändert)
def run_supervisor(state: SupervisorState) -> dict:
    """Der Supervisor-Knoten: Klassifiziert die Anfrage und setzt 'next'."""
    print("\n--- SUPERVISOR: Klassifiziere Anfrage... ---")
    
    prompt = (
        "Du bist ein Supervisor-Agent. Wähle EINES der verfügbaren Tools (tavily_search oder execute_code), "
        "ODER wähle KEIN Tool, wenn die Antwort generisch und einfach ist. Antworte in Deutsch."
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
        generic_prompt = f"Antworte kurz und direkt auf die Frage: {state['messages'][-1].content}. Antworte in Deutsch."
        final_response = supervisor_llm.invoke([HumanMessage(content=generic_prompt)])
        
        return {"messages": [final_response], "next": "end"}


def route_supervisor(state: SupervisorState) -> str:
    return state["next"]

# ====================================================================
# 3. GRAPH ERSTELLEN UND KOMPILIEREN (Basierend auf Ihrer Node-ID-Logik)
# ====================================================================

workflow = StateGraph(SupervisorState)

# Knoten hinzufügen (Wir verwenden die Node-IDs aus Ihrem funktionierenden Skript)
workflow.add_node("tavily_search", web_search_agent) 
workflow.add_node("execute_code", execute_code_agent) 
workflow.add_node("supervisor", run_supervisor) 

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
# 4. TESTFÄLLE
# ====================================================================

print("\n" + "="*50)
print("MULTI-AGENT SUPERVISOR START - ECHTE TOOLS")
print("="*50)

# Test 1: Web-Frage (Sollte Tavily wählen)
web_question = "Was ist das aktuelle meistverkaufte Buch in Deutschland?"
print(f"\n[TEST 1] FRAGE: {web_question}")
final_state_web = app.invoke({"messages": [HumanMessage(content=web_question)]}) 
print(f"FINALE ANTWORT: {final_state_web['messages'][-1].content}")


# Test 2: Code-Frage (Sollte Code wählen)
code_question = "Berechne 1530 geteilt durch 17"
print(f"\n[TEST 2] FRAGE: {code_question}")
final_state_code = app.invoke({"messages": [HumanMessage(content=code_question)]})
print(f"FINALE ANTWORT: {final_state_code['messages'][-1].content}")


# Test 3: Allgemeine Frage (Sollte END wählen)
general_question = "Nenne die Hauptfarbe eines Löwenzahns."
print(f"\n[TEST 3] FRAGE: {general_question}")
final_state_general = app.invoke({"messages": [HumanMessage(content=general_question)]})
print(f"FINALE ANTWORT: {final_state_general['messages'][-1].content}")
