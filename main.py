from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from rich.markdown import Markdown
from rich.console import Console
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

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
    console = Console()
    md = Markdown(response.content, code_theme="monokai", hyperlinks=True)
    # Print with rich formatting
    console.print(md)

def retrieval_with_lcel():
    retrieval_chain = (
            RunnablePassthrough.assign(context= itemgetter("question") | retriever | formatDocs)
            | prompt_template
            | llm
            | StrOutputParser()
        )
    return retrieval_chain

def main():
    print("Retrieving relevant documents...")
    question = "What is pinecone in machine learning?"
    # retrieval_without_lcel(question)
    retrieval_chain = retrieval_with_lcel()
    response = retrieval_chain.invoke({"question": question})
    print("Answer:")
    console = Console()
    md = Markdown(response, code_theme="monokai", hyperlinks=True)
    console.print(md)

def formatDocs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":    
    main()