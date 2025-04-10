import streamlit as st
import time
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from selenium import webdriver
import faiss
from selenium.webdriver.chrome.options import Options

from llama_index.core import (
    VectorStoreIndex,
    Document,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.query_engine import RetrieverQueryEngine

load_dotenv()

# === Gemini API ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing! Add it to the .env file.")

# === Web Scraper ===
def get_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    return webdriver.Chrome(options=options)

def extract_links(base_url, max_links=15):
    driver = get_driver()
    driver.get(base_url)
    time.sleep(2)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full_url = href if href.startswith("http") else base_url.rstrip("/") + "/" + href.lstrip("/")
        links.add(full_url)
        if len(links) >= max_links:
            break
    return list(links)

def scrape_docs(links):
    docs = []
    for url in links:
        try:
            driver = get_driver()
            driver.get(url)
            time.sleep(2)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            driver.quit()

            for tag in soup(["nav", "footer", "header", "script", "style"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            docs.append(Document(text=text, metadata={"source": url}))
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    return docs

# === Build Agent ===
def build_agent(docs):
    # Load embedding using new method
    print(f"\nLoaded {len(docs)} documents")
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    Settings.embed_model = embed_model

    faiss_index = faiss.IndexFlatL2(384)  # 384 = dimension of MiniLM embeddings
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    index = VectorStoreIndex.from_documents(docs, vector_store=vector_store)
    print(f"Vector index built with {len(index.docstore.docs)} documents.")
    retriever = index.as_retriever(similarity_top_k=4)
    query_engine = RetrieverQueryEngine(retriever=retriever)

    tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="product_help_docs",
            description="Answers questions about product features and integrations using help documentation."
        )
    )

    llm = Gemini(model="models/gemini-1.5-flash", api_key=GEMINI_API_KEY)

    agent = FunctionCallingAgent.from_tools(
        tools=[tool],
        llm=llm,
        system_prompt=(
            "You are an AI assistant for answering questions about help documentation. "
            "You have access to a tool called 'product_help_docs' and the query engine, which retrieves information from the docs. "
            "Use this tool whenever the user asks a question related to product features, usage, or integrations. "
        )
    )

    return agent

# === Streamlit UI ===
st.set_page_config(page_title="📘 AI Help Agent")
st.title("📘 AI Help Documentation Agent")

url = st.text_input("Enter Help Site URL (e.g., https://help.slack.com)")

if st.button("Index Website"):
    with st.spinner("Crawling and indexing help site..."):
        try:
            links = extract_links(url)
            documents = scrape_docs(links)
            agent = build_agent(documents)
            st.session_state.agent = agent
            st.success("Docs indexed and agent ready!")
        except Exception as e:
            st.error(f"Error: {e}")

if "agent" in st.session_state:
    question = st.text_input("Ask your question")
    if st.button("Ask"):
        with st.spinner("Thinking..."):
            response = st.session_state.agent.chat(question)
            st.markdown(f"### Answer:\n{response.response}")
