import requests
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from urllib.parse import urljoin, urlparse


def load_url_and_links(url: str):
    try:
        response = requests.get(url, timeout=10, verify=False)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.extract()
        text = soup.get_text(separator="\n", strip=True)
        # Extract all links
        links = set()
        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"])
            # Optional: filter to same domain
            if urlparse(link).netloc == urlparse(url).netloc:
                links.add(link)
        return Document(page_content=text, metadata={"source": url}), links
    except Exception as e:
        print(f"❌ Error loading {url}: {e}")
        return None, set()

def crawl_and_load(start_urls, max_pages=50):
    visited = set()
    to_visit = list(start_urls)
    documents = []
    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        doc, links = load_url_and_links(url)
        visited.add(url)
        if doc:
            documents.append(doc)
        # Add new links to queue
        for link in links:
            if link not in visited and link not in to_visit:
                to_visit.append(link)
    return documents


# Load and clean text from a single URL
def load_url(url: str):
    try:
        response = requests.get(url, timeout=10, verify=False)
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unnecessary tags
        for tag in soup(["script", "style"]):
            tag.extract()
        text = soup.get_text(separator="\n", strip=True)
        return Document(page_content=text, metadata={"source": url})
    except Exception as e:
        print(f"❌ Error loading {url}: {e}")
        return None

# Split documents into chunks
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)

# Create or persist vectorstore
def create_vectorstore(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="vectordb_urls")
    vectordb.persist()
    return vectordb

# Build RAG chain
def build_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = Ollama(model="mistral")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

URLS_TO_LOAD = [
    "https://docs.digital.ai/release/docs/category/get-started",
    "https://docs.digital.ai/release/docs/concept/core-concepts-of-xl-release",
    "https://docs.digital.ai/release/docs/glossary/release-glossary",
    "https://docs.digital.ai/release/docs/how-to/using-the-release-overview",
    "https://docs.digital.ai/release/docs/concept/release-life-cycle",
    "https://docs.digital.ai/release/docs/how-to/configure-release-properties",
    "https://docs.digital.ai/release/docs/category/phases",
    "https://docs.digital.ai/release/docs/category/tasks",
    "https://docs.digital.ai/release/docs/category/triggers",
    "https://docs.digital.ai/release/docs/category/variables",
    "https://docs.digital.ai/release/docs/concept/variables-in-xl-release"
    "https://docs.digital.ai/release/docs/how-to/create-a-jython-script-task",
    "https://docs.digital.ai/release/docs/how-to/create-release-variables",
    "https://docs.digital.ai/release/docs/xl-platform/how-to/work-with-xl-yaml-format-for-release",
    "https://docs.digital.ai/release/docs/how-to/using-the-xl-release-api-in-scripts",
    "https://docs.digital.ai/release/docs/how-to/create-a-groovy-script-task#variables-and-public-api-access"
]

# Main logic
def main():
    urls = URLS_TO_LOAD

    print("🔗 Loading URLs...")
    documents = []
    for url in urls:
        doc = load_url(url)
        if doc:
            documents.append(doc)

    if not documents:
        print("❌ No documents loaded. Exiting.")
        return
    
    # documents = crawl_and_load(URLS_TO_LOAD, max_pages=50)

    print(f"📄 Loaded {len(documents)} documents")
    chunks = split_documents(documents)
    print(f"✂️ Split into {len(chunks)} chunks")

    vectorstore = create_vectorstore(chunks)
    qa_chain = build_qa_chain(vectorstore)

    while True:
        query = input("\n🧠 Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        result = qa_chain({"query": query})
        print(f"\n🤖 Answer: {result['result']}")


if __name__ == "__main__":
    main()
