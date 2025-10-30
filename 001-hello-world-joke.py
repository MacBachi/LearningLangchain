# Importiert das spezifische LLM für OpenAI.
from langchain_openai import OpenAI 

# Import für die iBasis-Komponenten (PromptTemplate) für Eingabe-Vorlagen
from langchain_core.prompts import PromptTemplate 

# 1. Das LLM initialisieren. temperature=0.9 macht die Antwort kreativer und damit gut für Witze (SIC!)
# Temperatur:
#  Niedrig (z.B. 0.0 – 0.2) Deterministisch / Fokus. Faktenabfrage, Klassifikation, Datenextraktion, Code-Generierung, Übersetzungen.
#  Mittel (z.B. 0.3 – 0.7) Ausgewogen. Allgemeine Konversation, Zusammenfassungen, einfache Texterstellung.
#  Hoch (z.B. 0.8 – 1.0+) Kreativ / Zufällig. Brainstorming, kreatives Schreiben, Witze, Lyrik, Rollenspiele.
#  temperature=0.0: Immer die gleiche Antwort (solange Prompt und Kontext gleich sind).
#  temperature=0.9: Hohe Variation; du bekommst bei jedem Aufruf eine andere Antwort.
llm = OpenAI(temperature=0.9)

# 2. Einen Prompt Template definieren
# Definiert die Vorlage. {thema} ist die Variable, die später gefüllt wird.
# Erste Komponente der Kette. Sie nimmt die Eingabe (thema) entgegen.
prompt = PromptTemplate.from_template(
        "Schreibe einen kurzen, lustigen Witz über {thema}."
)

# 3. Die Chain mit LCEL erstellen
# Die Kette verbindet: Prompt -> LLM
# Die Ausgabe des prompt (der fertige Prompt-String) wird als Eingabe in das llm (das OpenAI-Modell) weitergeleitet.
chain = prompt | llm

# 4. Die Chain ausführen
# Führt die gesamte Kette aus. invoke() ist die standardmäßige synchrone Methode. 
# Das Input-Dictionary ({"thema": ...}) wird automatisch dem prompt übergeben.
#  chain.invoke(input)
#    Synchron. Blockiert, bis das Endergebnis vorliegt.
#    Standardaufruf für einzelne Anfragen.
#  chain.batch(inputs)
#    Synchron. Führt die Kette für eine Liste von Inputs aus.
#    Effiziente Verarbeitung mehrerer Anfragen in einem Durchgang.
#  chain.stream(input)
#    Synchron. Gibt einen Iterator zurück, der die Ergebnisse in Echtzeit (Token für Token) liefert.
#    Essentiell für eine schnelle Benutzererfahrung in Chat-Anwendungen.
#  chain.ainvoke(input)
#    Asynchron. await-fähige Version von invoke().
#    Bessere Performance in asynchronen Umgebungen (z.B. FastAPI, asyncio).
#  chain.astream(input)
#    Asynchron. await-fähige Version von stream().
#    Ermöglicht schnelles, asynchrones Streaming.
antwort = chain.invoke({"thema": "LangChain"}) 

print(antwort)

