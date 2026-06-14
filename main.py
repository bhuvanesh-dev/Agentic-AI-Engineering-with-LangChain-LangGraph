from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b"
)
vector_store = Chroma(
    collection_name="mediumblog1_collection",
    persist_directory="./chroma_langchain_db",
    embedding_function=embeddings,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question only based on the context:
    {context} 
    Question: {question}
    Provide a detailed answer with references to the context.
    """
)

llm = ChatOllama(model="qwen3.5:4b", temperature=0.7)


def retrieval_without_lcel(question):
    # Retrieval without LCEL (LangChain Expression Language)
    # Retrieve relevant documents based on the question
    relevant_docs = retriever.invoke(question)
    # Format the retrieved documents into a single string
    context = formatDocs(relevant_docs)
    # Create a prompt using the formatted context and the question
    message = prompt_template.format(context=context, question=question)
    # Generate a response using the LLM
    response = llm.invoke(message)
    print("Answer:")
    print(response.content)


def main():
    print("Retrieving relevant documents...")
    question = "What is pinecone in machine learning?"
    retrieval_without_lcel(question)

def formatDocs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":    
    main()