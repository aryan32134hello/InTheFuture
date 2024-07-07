from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
import os


def get_llm(model_id):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    llm = HuggingFaceEndpoint(repo_id=model_id, temperature=0.1, max_length=1024,timeout = 300)
    return llm


def website_loader(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    return data


def text_splitter_and_creating_db(data):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embedding_function)
    return db


def retriver(url, question, model_id):
    data = website_loader(url)
    db = text_splitter_and_creating_db(data)
    retriver = db.as_retriever(search_type="mmr", search_kwargs={'k': 4, 'fetch_k': 20})
    llm = get_llm(model_id)
    prompt = hub.pull("rlm/rag-prompt")
    ans = rag_chain(prompt, llm, question, retriver)
    return ans


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def rag_chain(prompt, llm, question, retriver):
    rag_chain = (
            {"context": retriver | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
    )
    print("invoking")
    ans = rag_chain.invoke(question)
    return ans


def main(url, question, model_id):
    ans = retriver(url, question, model_id)
    return ans
