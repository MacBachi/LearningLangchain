# 1. Daten laden (Simulierte Dokumente)
# Wir erstellen temporär eine kleine Textdatei für dieses Beispiel
with open("rag_data.txt", "w") as f:
    f.write("LangChain ist ein Framework zur Entwicklung von Anwendungen auf Basis von LLMs. ChromaDB ist ein Open-Source Vektorspeicher. Das beste Essen ist ein Wiener Schnitzel.")


#####################################################################################
# I. Indizierungsphase (Vorbereitung)
# Zuerst benötigen wir ein Dokument, das in unserem Vektorspeicher landet. 
#

# Importe für Indizierung und RAG-Kette (BEISPIELE!):
# Dateiformate
#   - PyPDFLoader
#   - CSVLoader
#   - UnstructuredFileLoader
# Web & APIs & Dienste
#   - WebBaseLoader
#   - RecursiveUrlLoader
#   - APILoader, JSONLoader
#   - YouTubeLoader, TwitterTweetLoader, WikipediaLoader, PubMedLoader
# Cloud & Datenbanken
#   - S3DirectoryLoader
#   - GCSDirectoryLoader
#   - PostgresLoader, SQLiteLoader
#   - AzureBlobStorageContainerLoader, GoogleDriveLoader, HuggingFaceDatasetLoader
# Notizen & Code
#   - EverNoteLoader, NotionLoader
#   - GitLoader, GitHubLoader
#   - GitLabLoader, BitbucketLoader, NotebookLoader
# Document Management
#   - ConfluenceLoader, SharePointLoader, BoxLoader
# Datenbanken
#   - CassandraLoader, MongoDBLoader, Neo4jLoader, ClickHouseLoader
# Messaging und Chat
#   - WhatsAppLoader, DiscordChatLoader, SlackDataLoader
# 
# Wichtig: Für viele dieser Loader (z.B. PyPDFLoader) musst du die spezifischen Abhängigkeiten zusätzlich installieren (poetry add pypdf).
# Es gibt Hunderte von Loadern, die LangChain in der langchain-community anbietet.
# Der beste Weg, sich einen Überblick zu verschaffen, ist die Betrachtung der Kategorien oder die offizielle Dokumentation.
# https://github.com/langchain-ai/langchain-community/tree/main/libs/community/langchain_community
# https://github.com/langchain-ai/langchain-community/tree/main/libs/community/langchain_community/document_loaders
#
# Weitere interessante Loader:
# assembly chromium epub excel obsidian pdf rss 
#
from langchain_community.document_loaders import TextLoader

# Allgemeine, einfache Splitter
#   - CharacterTextSplitter
#     - Zerlegt den Text anhand eines beliebigen Zeichens (\n\n, . oder ,).
#     - Der einfachste Splitter, nützlich für einfache Texte.
#   - RecursiveCharacterTextSplitter
#     - Der empfohlene Standard. Versucht, den Text zuerst anhand großer Trennzeichen zu splitten 
#       (z.B. \n\n), dann kleinerer (\n), dann noch kleinerer ( ), um die semantische Kohärenz zu erhalten.
#     - Besser für allgemeine Dokumente, um Absätze zusammenzuhalten.
# Semantisch & Dokumentenspezifisch
#   - SemanticChunker
#     - Nutzt Embeddings, um zu erkennen, wann sich das Thema ändert
#     - Für höchste semantische Qualität der Chunks, aber rechenintensiver.
#  - TokenTextSplitter
#     - Splittet basierend auf der Anzahl der Tokens eines bestimmten Modells (z.B. GPT).
#     - Wichtig, wenn Sie die maximale Token-Anzahl des LLMs streng einhalten müssen.
#  - MarkdownTextSplitter
#     - Zerlegt Markdown-Dateien unter Berücksichtigung von Überschriften, Listen und Codeblöcken.
#     - Für Dokumentation, READMEs und alle Markdown-basierten Texte.
#  - HTMLTextSplitter
# Fortgeschrittenes Chunking
#  - ParentDocumentRetriever: Speichert kleine Chunks für die Suche, 
#    ruft aber den gesamten übergeordneten Absatz (Parent) für den Kontext ab, 
#    um Halluzinationen zu reduzieren.
#  - CodeTextSplitter: Spezialisierte Splitter für verschiedene Programmiersprachen 
#    (z.B. PythonTextSplitter, JavaTextSplitter), die Code nach Klassen, Funktionen oder Methoden trennen.
#
# Zwei Parameter sind immer entscheidend:
#
# - chunk_size: 
#   Die maximale Größe des Chunks (z.B. 1000 Zeichen oder Tokens).
#
# - chunk_overlap: 
#   Wie viele Zeichen/Tokens am Ende eines Chunks im nächsten Chunk überlappen sollen. 
#   Dies ist wichtig, um den Kontext zu erhalten, wenn semantisch wichtige Informationen 
#   an einer Trennstelle liegen.
#
from langchain_text_splitters import CharacterTextSplitter


# Die OpenAIEmbeddings-Klasse ist der Schlüssel zur Umwandlung von Text in eine Sprache, 
# die der Vektorspeicher verstehen kann. Dies ist das Herzstück von RAG.
# Ein Embedding (Einbettung) ist eine numerische Vektor-Repräsentation eines Textstücks 
# (Wort, Satz, Dokument-Chunk).
#
# 1. Der Text Splitter liefert einen Dokument-Chunk.
# 2. OpenAIEmbeddings sendet diesen Chunk an die OpenAI API.
# 3. Die API gibt einen Vektor zurück (z.B. eine Liste von 1536 Fließkommazahlen).
# 4. Dieser Vektor wird zusammen mit dem ursprünglichen Text-Chunk im Vektorspeicher abgelegt.
#
# Alternativen (Andere Embedding-Modelle)
#
# OpenAIEmbeddings, HuggingFaceEmbeddings, GooglePalmEmbeddings / VertexAIEmbeddings
# 
from langchain_openai import OpenAIEmbeddings # NEU: Für Vektoren

# Die Klasse Chroma ist die Schnittstelle zum Vektorspeicher
# Ein Vector Store (Vektorspeicher) ist eine spezielle Datenbank, die darauf optimiert ist,
# Vektoren (die Zahlenreihen aus den Embeddings) und
# die zugehörigen Originaldaten effizient zu speichern und abzufragen.
# Er speichert Paare von:
#   - Dem numerischen Embedding-Vektor
#   - Dem zugehörigen Original-Text-Chunk.
# Er ermöglicht die "nächsten Nachbarn"-Suche (Nearest Neighbor Search). 
# Das heißt, er findet schnell die Vektoren, die dem Suchvektor der Benutzerfrage 
# im hochdimensionalen Raum am nächsten liegen.
# Chroma ist ein beliebter Open-Source-Vektorspeicher, 
# der einfach zu verwenden ist und sich gut für Entwicklungs- und kleinere Produktionsumgebungen eignet.
#
# Der eigentliche Vektorspeicher (vectorstore) wird selten direkt in der LCEL-Kette verwendet. 
# Stattdessen wandeln wir ihn in einen Retriever um, der ein standardisiertes LangChain-Interface ist.
#
from langchain_community.vectorstores import Chroma # NEU: Für Vektorspeicher

loader = TextLoader("rag_data.txt")
documents = loader.load()

# 1. Text aufteilen (Chunking)
# Die Wahl von chunk_size=1000 und chunk_overlap=0 sind Hyperparameter 
# und bestimmen maßgeblich die Leistung deines RAG-Systems. 🧐
#
# chunk-size=1000
#   - 1000 Zeichen (Beispielwert): 
#     Ist ein häufig gewählter, moderater Wert, der oft gut funktioniert. 
#     Er ist groß genug, um semantischen Kontext zu liefern, aber klein genug, 
#     um mehrere relevante Chunks in den LLM-Prompt zu packen.
#   - Je größer der Chunk: Desto mehr Kontext ist enthalten 
#     (weniger Gefahr, wichtige Infos zu trennen), 
#     aber das LLM muss mehr irrelevante Daten verarbeiten, 
#     was die Kosten erhöht und die Qualität verschlechtern kann 
#     (LLMs verlieren den Fokus bei sehr langen Kontexten).
#   - Je kleiner der Chunk: Desto präziser ist die Suche, 
#     aber Sie riskieren, dass semantisch zusammenhängende Sätze voneinander getrennt werden, 
#     was den Kontext zerstört.
# 
#   Faustregel: Beginnen Sie mit 500–1000 Zeichen und passen Sie an. 
#   Bei komplexen technischen Dokumenten könnte man auch bis zu 4000 Zeichen hochgehen.
#
# chunk_overlap=0 (Beispielwert):
#   Bedeutet keine Überlappung.
#   Das ist die einfachste Einstellung, aber in der Praxis oft schlecht für die RAG-Qualität.
#
#   Typische Werte: Man wählt oft eine Überlappung von 10-20% der Chunk-Größe 
#                   (z.B. bei chunk_size=1000 wäre chunk_overlap=100 bis 200).
#                   Dies stellt sicher, dass semantische Brücken an den Schnittstellen erhalten bleiben.
#
#  Für dieses initiale Beispiel wurde chunk_overlap=0 gewählt,
#  um das Konzept so einfach wie möglich zu halten, aber in realen Projekten 
#  sollten Sie eine Überlappung verwenden.
#
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 2. Einbetten und Speichern im Vector Store (Chroma)
# Wir verwenden OpenAIEmbeddings, um die Vektoren zu erzeugen
#
# Hier passiert die rechenintensive Vektorisierung Ihrer Dokumente, 
# die von einem externen Dienst (OpenAI) durchgeführt wird:
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# 3. Den Retriever erstellen
# Der Retriever ist die Komponente, die die relevanten Chunks abruft
# as_retriever():
#     Dieser Befehl erstellt lediglich eine Abfrageschnittstelle (den retriever), 
#     die weiß, wie man in dieser Datenbank sucht.
retriever = vectorstore.as_retriever()

# -> Bis zu der Zeile retriever = vectorstore.as_retriever() 
# -> ist die gesamte Vorbereitungs- oder Indizierungsphase abgeschlossen.
#     - Laden und Chunking
#     - Vektorisierung (API-Calls)
#     - Lokale Speicherung
#     - Erstellung Abfrageschnittstelle (den Retriever)
#
#  Das System ist nun abfragebereit, aber es wurde noch keine Frage gestellt oder beantwortet.
#  Der eigentliche RAG-Prozess zur Beantwortung der Frage startet erst mit dem Aufruf der Kette.


#####################################################################################
# II. LCEL RAG-Pipeline: Schritt für Schritt
# Jetzt verketten wir den Retriever mit dem Prompt und dem LLM mithilfe von LCEL.
#
#

# Im RAG-Flow ist ChatOpenAI die letzte Komponente, die die finale, menschenlesbare Antwort generiert
# Das Modell erhält den abgerufenen Kontext und die Benutzerfrage im Rahmen des RAG-Prompts 
# und liefert die zusammenfassende Antwort.
from langchain_openai import ChatOpenAI

# Methode zur Definition von Prompts für Chat Models (wie ChatOpenAI)
# Im Gegensatz zum einfachen PromptTemplate (den wir für den Witz verwendet haben, 
# der nur einen String liefert, generiert der ChatPromptTemplate eine Liste von Nachrichten (Messages).
#
# - *System Message* (System-Prompt):
#     Definiert die Rolle, Regeln oder den Stil des Modells 
#     (z.B. "Du bist ein hilfreicher Assistent, der nur basierend auf dem Kontext antwortet.").
# - *Human Message* (Benutzer-Prompt): 
#     Enthält die Frage des Benutzers und den Kontext.
# - *AIMessage* (Assistent-Antwort):
#     Enthält die vorherigen Antworten des Modells (für den Chat-Verlauf).
#
# Im RAG-Kontext ist der ChatPromptTemplate entscheidend für die Anweisung des Modells
#
# Die Verwendung von ChatPromptTemplate ermöglicht:
# - Rollenbasierte Prompts: Präzisere Kontrolle über das Modellverhalten.
# - Bessere Leistung: Moderne LLMs sind auf dieses Message-Format optimiert.
#
from langchain_core.prompts import ChatPromptTemplate

# RunnablePassthrough ist ein wichtiges Konzept in der LangChain Expression Language (LCEL) 
# und fungiert als Datenverteiler in der Kette.
#
# Es ist eine spezielle LCEL-Komponente, die ihren Input unverändert an ihren Output weitergibt 
# und gleichzeitig andere Operationen (wie den retriever) ausführen lässt.
#
# Es ist eine Art Daten-Tunnel: Alles, was reinkommt, kommt auch wieder raus, 
# während ein "Side-Task" ausgeführt wird.
#
# In der RAG-Chain wird sie verwendet, um sicherzustellen, 
# dass die ursprüngliche Benutzerfrage (question) sowohl an den Retriever als auch an den Prompt 
# übergeben werden kann.
#
from langchain_core.runnables import RunnablePassthrough

# Der StrOutputParser ist oft die letzte Komponente in einer LCEL-Kette
# deren Aufgabe es ist, die rohe Ausgabe des LLM in einen einfachen Python-String umzuwandeln.
#
# Wenn ein Chat Model (wie ChatOpenAI) in LangChain aufgerufen wird, 
# gibt es keine einfache Zeichenkette zurück, sondern ein komplexeres Objekt: eine AIMessage-Instanz.
#
# Die AIMessage enthält Metadaten (wie Rolleninformationen und Beendigungsgründe) 
# zusätzlich zum eigentlichen Text.
#
# Andere Parser:
# - JsonOutputParser
# - PydanticOutputParser
# - ...
#
from langchain_core.output_parsers import StrOutputParser

# 1. Prompt-Template für RAG
# Der Kontext {context} ist ZWINGEND erforderlich, damit das LLM die abgerufenen Chunks nutzt.
#
# Dieser Template hat die kritische Aufgabe, dem LLM klare Anweisungen zu geben 
# und den abgerufenen Kontext sauber von der Frage zu trennen.
#
# Der Template, den du definierst, wird intern von ChatPromptTemplate in eine Liste von Nachrichten (Messages) 
# umgewandelt. Typischerweise wird das gesamte Template zu einer einzigen 
#  - HumanMessage (Benutzer-Nachricht) kombiniert, oder die erste Anweisung wird in eine separate 
#  - SystemMessage (System-Anweisung) extrahiert, um dem LLM die Regeln vorzugeben.
#
# Die Zeile "Antworte basierend nur auf dem folgenden Kontext:"
# ist der wichtigste Teil des System-Promptings im RAG. Sie dient als Guardrail (Schutzschiene)
#
# Sie weist das LLM an, seine internen Trainingsdaten (sein Allgemeinwissen) zu ignorieren.
# Dies minimiert Halluzinationen und stellt sicher, 
# dass die Antwort faktisch nur auf deinen Quelldokumenten basiert.
#   - {context}
#     Das Ergebnis des retriever (die relevanten Text-Chunks)
#   - {question}
#     Die ursprüngliche Eingabe des Benutzers, die durch RunnablePassthrough() weitergereicht wird.
#
# Der {context} ist die stark verkürzte, hochrelevante Version der Ausgangsdokumente, 
# die für die Beantwortung der aktuellen Frage ausgewählt wurde.
#
template = """Antworte basierend nur auf dem folgenden Kontext:
{context}

Frage: {question}"""

prompt = ChatPromptTemplate.from_template(template)

# 2. Das LLM initialisieren (Chat-Modell für bessere Leistung bei RAG)
# Andere Modelle zB:
#   - gpt-4o
#   - gpt-4o-mini
#   - gpt-4-turbo
#   - gpt-3.5-turbo
# Es können aber auch andere Modelle anderer Anbieter (zB. Google, Anthropic, Meta, Hugging Face).
# Um diese Modelle zu verwenden, müssen Sie das entsprechende anbieterspezifische Paket installieren 
# (poetry add langchain-google-genai) und den zugehörigen API-Schlüssel setzen.
# 
# Für den Umstieg auf ein anderes Modell müssten Sie in Ihrem Code nur die Importzeile 
# und die Initialisierung anpassen, da alle die gleiche standardisierte Chat Model-Schnittstelle 
# in LangChain nutzen.
# 
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 3. Die RAG-Kette mit LCEL erstellen
rag_chain = (
    # RunnablePassthrough sorgt dafür, dass das Input-Dictionary (question)
    # an die nächste Komponente weitergegeben wird.
    #
    # Das ist der Punkt in der LCEL-Kette, an dem die Inhalts- und Datenzuweisung stattfindet, 
    # um den Prompt für das LLM zu füllen.
    #
    # -"context": retriever
    #    - Die 'frage' wird als Input an den retriever gesendet.
    #    - Der Retriever führt die Vektorsuche (lokal in ChromaDB) durch.
    #    - Das Ergebnis – die relevanten Text-Chunks – wird dem Schlüssel "context" zugewiesen.
    # - "question": RunnablePassthrough()
    #    - Die frage wird an den RunnablePassthrough gesendet.
    #    - Dieser gibt die ursprüngliche frage unverändert weiter.
    #    - Die frage selbst wird dem Schlüssel "question" zugewiesen.
    #
    #  Das Ergebnis sollte dann sein:
    #  {
    #    "context": ["Das beste Essen ist der Wiener Schnitzel."],  # Gefüllt vom Retriever
    #    "question": "Was ist das beste Essen, laut den Dokumenten?" # Gefüllt vom Passthrough
    #  }
    #
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt # Der Prompt bekommt jetzt {context} vom Retriever und {question} vom Input
    | llm
    | StrOutputParser() # Output Parser, um eine reine String-Antwort zu erhalten
)

# 4. Die Kette ausführen
frage = "Was ist das beste Essen, laut den Dokumenten?"
antwort = rag_chain.invoke(frage)

print(f"Frage: {frage}")
print(f"Antwort: {antwort}")
