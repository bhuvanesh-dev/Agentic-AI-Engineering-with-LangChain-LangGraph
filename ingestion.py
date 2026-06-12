from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4


def main():
    load_dotenv()

    print("Ingestion process started...")

    loader = TextLoader("mediumblog1.txt", encoding="utf-8")
    documents = loader.load()

    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Number of chunks created: {len(chunks)}")

    embeddings = OllamaEmbeddings(
        model="qwen3-embedding:0.6b"
    )

    print("Adding chunks to Chroma vector store...")

    
    vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
    )

    uuids = [str(uuid4()) for _ in range(len(chunks))]

    vector_store.add_documents(documents=chunks, ids=uuids)

    print("Ingestion process completed successfully.")

    results = vector_store.similarity_search_by_vector(
        embedding=embeddings.embed_query("How similarity search works?"), k=1
    )
    for doc in results:
        print(f"* {doc.page_content} [{doc.metadata}]")

if __name__ == "__main__":
    main()