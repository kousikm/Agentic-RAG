from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from ingest import get_vectorstore

def create_rag_chain(api_key=None):
    """Creates a RAG chain using the vector store."""
    vectorstore = get_vectorstore(api_key)
    if not vectorstore:
        return None

    retriever = vectorstore.as_retriever()
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

def query_agent(question: str, api_key=None):
    """Queries the RAG agent."""
    rag_chain = create_rag_chain(api_key)
    if not rag_chain:
        return "Vector store not found. Please ingest documents first."
    
    response = rag_chain.invoke({"input": question})
    return response["answer"]
