from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import argparse

def main():
    parser = argparse.ArgumentParser(description="Argumentos")
    parser.add_argument("-name", type=str, required=True, help="Nombre del documento .pdf")
    args = parser.parse_args()
    print(f"Cargando documento: {args.name}")
    embedding = OllamaEmbeddings(
        model="llama3.1",
        )

    loader = PyPDFLoader(f"{args.name}")
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(pages)

    vector_store = Chroma.from_documents(
        documents=all_splits,
        embedding=embedding,
        persist_directory="./mi_vectorstore"
    )

    vector_store.persist()
    print("Vectorstore guardado en ./mi_vectorstore")

if __name__ == "__main__":
    main()