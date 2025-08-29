import os
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

from langchain import hub
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import asyncio
from langchain_community.vectorstores import Chroma
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

async def load_docs():
    print("Loading docs...")
    file_path=os.path.join("resumen3.pdf")
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages


def create_graph(vector_store,llm):
    print("Creating graph...")
    prompt = hub.pull("rlm/rag-prompt")

    sys_msg = """Eres un asistente que ayuda a definir que esta pidiendo el usuario. Esta petición puede ser una de las siguientes:
    1. Question: El usuario esta preguntando sobre un concepto de matemáticas.
    2. Generate: El usuario esta pidiendo que se genere un documento de ejercicios o explicativo.
    3. Nothing: El usuario esta pidiendo algo que no tiene que ver con conceptos matematicos.

    Responde solo con el nombre de la opción que corresponde (Question, Generate o Nothing) y nada más.
    """

    nothing_message = "Eres un asistente que ayuda a responder preguntas generales que no tienen que ver con conceptos matemáticos. Explicame brevemente porque mi petición no tiene que ver con conceptos matemáticos."
    generate_message = "El usuario esta pidiendo generar una guia de estudio. Dale un texto detallado que contenga las definiciones de los conceptos claves, algunos ejemplos y ejercicios, incluye cualquier detalle que el usuario haya pedido."

    sys_msg = SystemMessage(content=sys_msg)
    hm_nothing_msg = HumanMessage(content=nothing_message)
    hm_generate_msg = HumanMessage(content=generate_message)

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    def options(state: State):
        x={"answer": [llm.invoke([sys_msg] + [state["question"]])]}
        print("options: ",x["answer"][-1].content)
        return x

    def decide_node(state: State)-> Literal["Question", "Generate", "Nothing"]:
        if "Question" in state["answer"][-1].content:
            return "Question"
        elif "Generate" in state["answer"][-1].content:
            return "Generate"
        else:
            return "Nothing"
        
    def nothing_node(state: State):
        x={"answer": llm.invoke([state["question"]] + [hm_nothing_msg]).content}
        print("nothing: ",x["answer"])
        return x

    def question_node(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages)
        print("question: ",response.content)
        return {"answer": response.content, "context": retrieved_docs}

    def generate_node(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
        messages = prompt.invoke({"question": state["question"], "context": docs_content}).to_messages()
        print("messages: ",messages, "hm_generate_msg: ",hm_generate_msg)
        response = llm.invoke(messages + [hm_generate_msg])
        string_to_pdf(response.content, "guia_estudio.pdf")
        return {"answer": "El documento se ha generado en guia_estudio.pdf", "context": retrieved_docs}


    # Compile application and test
    graph_builder = StateGraph(State)
    graph_builder.add_node("options",options)
    graph_builder.add_node("Nothing",nothing_node)
    graph_builder.add_node("Question",question_node)
    graph_builder.add_node("Generate",generate_node)

    graph_builder.add_edge(START, "options")
    graph_builder.add_conditional_edges("options",decide_node)
    graph_builder.add_edge("Question", END)
    graph_builder.add_edge("Generate", END)

    within_thread_memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=within_thread_memory)
    app(graph)


def string_to_pdf(text: str, filename: str):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    for line in text.split("\n"):
        if line.strip():  
            story.append(Paragraph(line, styles["Normal"]))
        story.append(Spacer(1, 12))  

    doc.build(story)

def app(graph):
    config={"configurable": {"thread_id": 0, "user_id": 0}}

    st.set_page_config(page_title="Chat Demo", layout="centered")

    st.title("Chat con Documentos de Matemáticas")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**Tú:** {msg['content']}")
        else:
            st.markdown(f"**Bot:** {msg['content']}")

    question = st.chat_input("Escribe tu mensaje...")

    if question:
      
        st.session_state.messages.append({"role": "user", "content": question})

        answer = graph.invoke({"question": question},config)['answer']

        st.session_state.messages.append({"role": "bot", "content": answer})

        st.rerun()

async def main():
    print("Starting main...")
    embedding = OllamaEmbeddings(
    model="llama3.1",
    )

    llm = ChatOpenAI(
        api_key="ollama",
        model="llama3.1",
        base_url="http://localhost:11434/v1"
    )

    vector_store = Chroma(
    persist_directory="./mi_vectorstore",
    embedding_function=embedding
    )

    create_graph(vector_store,llm)

if __name__ == "__main__":
    asyncio.run(main())