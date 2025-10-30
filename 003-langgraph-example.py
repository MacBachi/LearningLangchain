"""
# Das Python-Modul operator bietet effiziente Funktionen, 
# die den Standardoperatoren von Python entsprechen 
# (z.B. operator.add entspricht +, operator.mul entspricht *).
#
# Seine Rolle in LangGraph: 
# Es wird verwendet, um das Reduktionsverhalten (Aggregationsverhalten) eines Zustandskanals zu definieren.
"""
import operator

"""
# standardmäßiges Python und nicht spezifisch für LangChain.
# Diese Imports stellen sicher, dass Ihr Code typisiert und lesbar 
# ist und dass LangGraph die notwendigen Aggregationsanweisungen erhält, um den Zustand korrekt zu verwalten.
# - TypedDict
#     Ermöglicht die Erstellung von Dictionary-Klassen mit vordefinierten Schlüsseln und deren Typen.
#     Definiert das feste Schema des Agenten-Speichers (z.B. muss der Schlüssel messages existieren).
# - Annotated
#     Ermöglicht es, einem Typ (wie List) zusätzliche Metadaten oder Anweisungen beizufügen.
#     Dient dazu, LangGraph die Reduktionslogik mitzuteilen (nämlich operator.add).
# - List
#     Definiert, dass eine Variable eine Liste von Elementen ist.
#     Wird verwendet, um den Typ des Chat-Verlaufs als List[AnyMessage] zu definieren.
"""    
from typing import TypedDict, Annotated, List

"""
# Core LangChain Komponenten
# Dieser Import bindet das leistungsstärkste und am häufigsten genutzte Modell-Interface
# von LangChain in dein Skript ein. 
#
# - langchain_openai
#     Dies ist das anbieterspezifische Paket für alle OpenAI-Integrationen
#     (LLMs, Embeddings). LangChain gliedert diese aus dem Kern-Framework aus, um die Basis klein zu halten.
# - ChatOpenAI
#     Stellt eine standardisierte Schnittstelle (Wrapper) für die modernen Chat-Endpunkte von OpenAI 
#     (wie GPT-4o, GPT-4 und GPT-3.5-Turbo) bereit.
#
# In LangChain wird ChatOpenAI gegenüber der älteren OpenAI-Klasse bevorzugt, 
# weil es direkt mit dem Konzept von Nachrichtenlisten (Messages) arbeitet.
#
# Im Kontext deines LangGraph-Agenten ist ChatOpenAI der zentrale Baustein im Knoten 
# call_llm, der für die gesamte Argumentation und Antwortgenerierung verantwortlich ist.
"""
from langchain_openai import ChatOpenAI

"""
# Die Imports HumanMessage und AIMessage sind elementar für jeden LangGraph-Agenten 
# und jedes moderne Chatbot-System in LangChain. Sie definieren, wer was im Gespräch gesagt hat.
#
# Moderne LLMs (wie GPT-4o) verarbeiten keine langen Textblöcke mehr,
# sondern eine Liste von Nachrichten (messages), wobei jeder Nachricht eine Rolle zugewiesen wird.
# Dies ist essenziell für die Aufrechterhaltung des Kontexts.
#
# - HumanMessage
#     Rolle: Definiert Text, der vom Benutzer oder der Anwendung kommt
#     und eine Aktion oder Antwort vom Agenten erfordert.
# - AIMessage
#     Rolle: Definiert Text, der vom Künstliche-Intelligenz-Modell (AI, Assistant) generiert wurde.
#
# Durch die Verwendung dieser getrennten Klassen (und der übergeordneten Klasse 
# AnyMessage in AgentState) kann der Agent:
#   - Den Verlauf verwalten: Die messages-Liste im Zustand ist eine 
#     chronologische Abfolge von HumanMessage und AIMessage.
#   - Entscheidungen treffen: Im Router (decide_next_step) kannst du
#     gezielt auf die letzte AIMessage zugreifen, um die Entscheidung (z.B. "Pizza gefunden?") zu treffen.
#
# Diese Nachrichten-Typen sind die Grundlage dafür, dass dein LangGraph weiß, was gerade im Dialog passiert ist.
"""
from langchain_core.messages import HumanMessage, AIMessage

"""
# LangGraph Komponenten
# internes Werkzeug von LangGraph, das die Grundlage für den Zustand (State) deines Agenten bildet.
# AnyMessage dient als Platzhalter oder Basis-Klasse für alle spezifischen 
# Nachrichtentypen in LangChain/LangGraph, insbesondere HumanMessage und AIMessage.
#
# Durch die Verwendung von AnyMessage wird dem Zustandsschema mitgeteilt,
# dass die Liste messages Elemente enthalten kann, die entweder vom Typ HumanMessage oder AIMessage sind.
#
# In LangGraph wird dieser generische Typ bevorzugt, weil er auch andere spezielle
# Nachrichten-Typen abdecken kann (z.B. SystemMessage oder ToolMessage, die wir 
# später bei der Tool-Integration sehen werden).
#
# Zusammenfassend: AnyMessage ist der Grundtyp in LangGraph,
# der es dir ermöglicht, einen gemischten Chat-Verlauf (User und AI) im Zustand zu speichern.
"""
from langgraph.graph.message import AnyMessage

"""
# Diese Imports stammen direkt aus dem Kernpaket langgraph.graph und 
# definieren die Struktur und den Abschluss deines Agenten-Workflows.
#
# - StateGraph
#     zentrale Klasse in LangGraph. Sie wird verwendet, um den gesamten Agenten-Workflow 
#     zu definieren und zu bauen. Du initialisierst sie mit deinem 
#       - Zustand (AgentState) und fügst dann 
#       - Knoten (add_node) und 
#       - Kanten (add_edge/add_conditional_edges) hinzu.
# - END
#     Dies ist eine spezielle, eingebaute Konstante.
#     Sie dient als Zielknoten in einer bedingten Kante, um den Graphen an dieser Stelle sauber zu beenden.
#
"""
from langgraph.graph import StateGraph, END # END ist die saubere Konstante für das Ende

"""
# ====================================================================
# 1. ZUSTAND DEFINIEREN (AgentState)
# ====================================================================
#
# Definiert das Schema unseres Graphen-Zustands
"""
class AgentState(TypedDict):

    """
    # 'messages' speichert den Chat-Verlauf. operator.add hängt neue Messages an.
    # 'messages' ist der Zustandskanal (der Speicher für den Chat-Verlauf).
    #
    # operator.add teilt LangGraph mit: 
    # "Wenn ein Knoten ein Update für messages zurückgibt, hänge es an die bestehende 
    # Liste an (list.extend oder list.__add__), anstatt die gesamte Liste zu überschreiben."
    #
    # Ohne diesen Import und diese Annotation würde das System bei jedem LLM-Aufruf versuchen,
    # die gesamte Historie zu überschreiben, und der Agent würde den vorherigen Kontext vergessen.
    # import operator und die Verwendung von operator.add stellen sicher,
    # dass Ihr Agent ein Gedächtnis (Memory) über mehrere Iterationen hinweg behält.
    """
    messages: Annotated[List[AnyMessage], operator.add]

"""
# ====================================================================
# 2. KNOTEN-FUNKTION (NODE)
# ====================================================================
#
# Initialisiere das LLM
"""
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

"""
# Knotendefinition: Ruft das LLM auf
#
# Diese Funktion ist als Knoten im LangGraph registriert und wird immer dann ausgeführt, 
# wenn die Ausführung den start_node erreicht.
#
# Input (state): Der Knoten erhält den aktuellen Zustand des Graphen. 
# Das ist entscheidend, denn nur so greift der Agent auf seinen gesamten Chat-Verlauf (state["messages"]) zu.
#
# Zusammenfassend: call_llm nimmt den gesamten Konversationskontext, 
# lässt das LLM die nächste Antwort generieren und fügt diese Antwort dann dem Speicher des Agenten hinzu, 
# bevor die Kontrolle an den nächsten Schritt (die Routing-Logik) weitergegeben wird.
"""
def call_llm(state: AgentState):
    """Führt den LLM-Aufruf durch und aktualisiert den Zustand."""
    print("--- NODE: LLM wird aufgerufen...")

    """
    # Der Chat-Verlauf wird an das LLM gesendet
    # 
    # llm.invoke(state["messages"]): Dies ist der eigentliche Aufruf des LLM (ChatOpenAI). 
    # Es übergibt die gesamte Historie der Konversation (die Liste der HumanMessage und AIMessage) an das Modell.
    #
    # Das LLM liest den Verlauf und generiert die nächste logische Antwort.
    """
    result = llm.invoke(state["messages"])
    
    """  
    # Aktualisiert den Zustand: Fügt die LLM-Antwort hinzu (wird durch operator.add verwaltet)
    # 
    # Rückgabe als Dictionary: Wie in LangGraph vorgeschrieben,
    # muss der Knoten ein Dictionary zurückgeben, um den Zustand zu aktualisieren.
    # 
    # Aktualisierung ({"messages": [result]}):
    #   - Der Schlüssel messages entspricht dem Zustandskanal in AgentState.
    #   - [result] ist die neue AIMessage vom LLM.
    #   - LangGraph verwendet das definierte Reduktionsverhalten 
    #     (operator.add in AgentState), um diese neue Nachricht zum 
    #     bestehenden Chat-Verlauf hinzuzufügen.
    """
    return {"messages": [result]}
"""
# ====================================================================
# 3. ROUTER-FUNKTION (BESTIMMT DIE KANTE)
# ====================================================================
#
# Diese Funktion ist NICHT als Node registriert, sondern steuert die bedingte Kante
#
# Dies ist die Router-Logik Ihres Graphen, die den Ablauf steuert.
#
# Diese Funktion wird nicht als eigenständiger Knoten ausgeführt, sondern als Entscheidungslogik 
# innerhalb der bedingten Kante (add_conditional_edges). Ihre Aufgabe ist es, zu prüfen, ob der 
# Agent seine Aufgabe abgeschlossen hat, und dann den Namen des nächsten gewünschten Knotens 
# (oder END) als String zurückzugeben.
#
# Wie jeder Knoten oder jede Routing-Funktion in LangGraph erhält sie den aktuellen Zustand (state) als Input. 
# Sie liest den Zustand, ohne ihn zu verändern.
#
# Zusammenfassend: decide_next_step ist das Gehirn des Graphen.
# Es ist die Kontrolllogik, die dem LangGraph mitteilt, ob der aktuelle Schleifen-Zyklus 
# abgeschlossen ist oder ob der Agent einen weiteren Schritt ausführen muss.
"""
def decide_next_step(state: AgentState):
    """
    # Entscheidet, ob der Graph beendet (END) oder fortgesetzt (Loop) werden soll.
    #
    # Ruft die letzte Nachricht (die AI-Antwort) ab
    #   last_message = state["messages"][-1]: 
    #     Die Funktion greift auf die Liste der Nachrichten zu und holt das letzte Element heraus. 
    #     Da der Router direkt nach dem call_llm-Knoten läuft, ist dies immer die neueste 
    #     AIMessage des Sprachmodells.
    #
    # Hier befindet sich die eigentliche Geschäftslogik. 
    # Im Minimalbeispiel prüfen wir, ob das Wort "pizza" in der Antwort enthalten ist. 
    # In einem realen Agenten würde hier oft eine komplexere Logik stehen, 
    # z.B. die Analyse eines Tools-Aufrufes, der vom LLM vorgeschlagen wurde.
    """
    last_message = state["messages"][-1]
    
    """
    # Der String-Output (Routing)
    #
    # - Die Funktion gibt einen String zurück ("end" oder "continue").
    # - Dieser String ist der Schlüssel im add_conditional_edges-Dictionary.
    #
    # 
    """
    # Prüfe die letzte AI-Antwort auf das Wort "pizza"
    if "pizza" in last_message.content.lower():
        print("--- ROUTER: Schlüsselwort 'pizza' gefunden. ENDE.")
        return "end" # String, der dem END-Knoten zugeordnet wird
    else:
        print("--- ROUTER: Kein Schlüsselwort gefunden. LOOP.")
        return "continue" # String, der dem 'start_node' zugeordnet wird

# ====================================================================
# 4. GRAPH ERSTELLEN UND KOMPILIEREN
# ====================================================================

"""
# Graph initialisieren
# Dies initialisiert den Graphen.
# Er weiß nun, dass sein Zustand das von uns definierte AgentState-Schema verwenden muss
# (mit dem messages-Speicher).
"""
workflow = StateGraph(AgentState)

"""
# Knoten hinzufügen (Wir brauchen nur den LLM-Knoten)
#
# workflow.add_node("start_node", call_llm):
#   Registriert die Funktion call_llm als einen Knoten im Graphen.
#   Der String "start_node" ist der eindeutige Name, mit dem wir diesen Knoten später referenzieren.
"""
workflow.add_node("start_node", call_llm)

"""
# Zweck: Definiert den ersten Knoten, der ausgeführt wird,
# wenn der kompilierte Graph (app.invoke()) aufgerufen wird. 
# Die Ausführung des Workflows beginnt immer hier.
#
# 1. Startpunkt definieren
"""
workflow.set_entry_point("start_node")

"""
# Dies ist der wichtigste Teil, da er die Schleifenlogik etabliert:
# 
#   - "start_node" (QUELLE):
#       Dies definiert den Knoten, nachdem die Funktion call_llm beendet wurde. 
#       Die Logik muss entscheiden, wohin es als Nächstes gehen soll.
#   - decide_next_step (REGEL): 
#       Dies ist die Routing-Funktion (das Gehirn). 
#       LangGraph ruft diese Funktion mit dem aktuellen Zustand auf. 
#       Sie muss einen String zurückgeben ("end" oder "continue").
#   - Mapping ({...}):
#       - "end": END: Wenn die Regel "end" zurückgibt (Pizza gefunden), 
#         beendet die Kante den gesamten Graphen am eingebauten END-Punkt.
#       - "continue": "start_node": Wenn die Regel "continue" zurückgibt 
#         (keine Pizza gefunden), springt die Kante zurück zum Knoten "start_node". 
#         Dies erzeugt die zyklische Schleife.
#
# 2. Bedingte Kanten (beginnen nach dem LLM-Aufruf)
"""
workflow.add_conditional_edges(
    "start_node",           # QUELLE ist der LLM-Knoten
    decide_next_step,       # Funktion, die den nächsten Schritt bestimmt (gibt String zurück)
    {
        "end": END,         # Wenn Funktion "end" zurückgibt -> ENDE des Graphen
        "continue": "start_node" # Wenn Funktion "continue" zurückgibt -> Gehe zurück zum Start (Loop)
    }
)

# Den Graphen kompilieren
app = workflow.compile()

# ====================================================================
# 5. AUSFÜHRUNG
# ====================================================================

# Starten des Graphen mit der ersten Benutzernachricht
initial_message = HumanMessage(content="Erzähl mir etwas über LangChain und wie man es benutzt, versuche aber auch, das Wort 'Pizza' einzubauen.")

print("\n--- STARTE GRAPH ---")
final_state = app.invoke({"messages": [initial_message]})

# Ausgabe der finalen Antwort
print("\n--- FINALE ANTWORT ---")
final_response = final_state['messages'][-1].content
print(f"Agenten-Antwort:\n{final_response}")
