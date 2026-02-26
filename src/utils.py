import os
import logging
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

logger.info("OpenAI API key loaded successfully.")


def load_data(file_path): 
    """Load data from a text file.""" 
    loader = TextLoader(file_path, encoding = "utf-8") 
    documents = loader.load() 
    print(f"Loaded {len(documents)} documents from {file_path}") 
    return documents


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return text_splitter.split_documents(text)

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY
    )

    vector_store = Chroma.from_documents(
        text_chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    if not vector_store:
        logger.error("Failed to create vector store.")
        raise ValueError("Vector store creation failed.")
    logger.info("Vector store created successfully.")
    vector_store.persist()
    return vector_store

def build_qa_chain(vector_store, k=3, model="gpt-4o-mini", temperature=0.2, verbose=False):
    """
    Build a RetrievalQAWithSourcesChain using a Chroma vector store.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=OPENAI_API_KEY
    )

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=verbose,
    )

    logger.info("QA chain built successfully.")
    return qa_chain


def ask(qa_chain, question: str, show_snippets: bool = False, snippet_len: int = 250):
    """
    Run a question through the QA chain and return (answer, sources, source_documents).
    """
    result = qa_chain.invoke({"question": question})

    answer = result.get("answer", "")
    sources = result.get("sources", "")
    source_docs = result.get("source_documents", [])

    if show_snippets and source_docs:
        for i, doc in enumerate(source_docs, 1):
            snippet = doc.page_content[:snippet_len].strip()
            logger.info(f"Doc {i} snippet: {snippet}")

    return answer, sources, source_docs