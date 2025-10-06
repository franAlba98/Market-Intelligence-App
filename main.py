import requests
from requests_html import HTMLSession
from bs4 import BeautifulSoup
import json

# Langchain Components (from your notebooks)
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# LangSmith for browser visualization
import os
from langsmith import Client

BASE_URL = "https://ir.tesla.com"
PRESS_URL = BASE_URL + "/press"

# LangSmith Configuration (for browser visualization)
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = "tesla-rag-system"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"

# Set up LangSmith tracing (only if API key is provided)
if LANGCHAIN_API_KEY != "YOUR_LANGSMITH_API_KEY_HERE":
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    print("🔗 LangSmith tracing enabled - view your chains at https://smith.langchain.com")
else:
    print("LangSmith API key not set - chains won't be traced to browser")

def scrape_press_releases(num_docs=5):
    """
    Scrape press releases from Tesla's investor relations page
    If scraping fails, returns sample data for development
    """
    try:
        session = HTMLSession()
        res = session.get(PRESS_URL)
        res.html.render(timeout=20)  # runs headless Chromium to execute JS

        print("Scraping Tesla press releases...")
        soup = BeautifulSoup(res.text, "html.parser")

        # Each press release is in a div with class "views-row"
        links = soup.select("div.views-row a")
        docs = []
        
        if not links:
            print("No press release links found.")
            return []
        
        print(f"Found {len(links)} press release links")
        
        for link in links[:num_docs]:
            url = BASE_URL + link.get("href")
            page = requests.get(url)
            page_soup = BeautifulSoup(page.text, "html.parser")

            # Title is in h1
            title_el = page_soup.select_one("h1")
            title = title_el.get_text(strip=True) if title_el else "Untitled"

            # Date is inside div with class "dateline"
            date_el = page_soup.select_one(".dateline")
            date = date_el.get_text(strip=True) if date_el else "Unknown"

            # Body content paragraphs inside ".field-item"
            content = " ".join([p.get_text(strip=True) for p in page_soup.select(".field-item p")])

            docs.append({
                "title": title,
                "date": date,
                "url": url,
                "content": content
            })

        return docs
        
    except Exception as e:
        print(f"Scraping failed: {e}")
        return []


def fetch_tesla_sec_filings(max_docs=5):
    """
    Fetch recent Tesla SEC filings (10-K, 10-Q, 8-K) using SEC submissions API.
    No fabricated data; pulls real filings and downloads their primary document.
    """
    try:
        # Tesla CIK without leading zeros
        cik_no_zeros = "1318605"
        submissions_url = f"https://data.sec.gov/submissions/CIK000{cik_no_zeros}.json"
        headers = {
            "User-Agent": "Tesla RAG App contact@example.com",
            "Accept-Encoding": "gzip, deflate",
        }
        resp = requests.get(submissions_url, headers=headers, timeout=20)
        if resp.status_code != 200:
            print(f"SEC submissions fetch failed: {resp.status_code}")
            return []
        data = resp.json()
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        primaries = recent.get("primaryDocument", [])
        filing_dates = recent.get("filingDate", [])

        wanted = {"10-K", "10-Q", "8-K"}
        docs = []
        for form, acc, primary, fdate in zip(forms, accessions, primaries, filing_dates):
            if len(docs) >= max_docs:
                break
            if form not in wanted:
                continue
            # Build archive URL: /edgar/data/{cik}/{accession_no_no_dashes}/{primary}
            acc_nodash = acc.replace("-", "")
            archive_url = f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{acc_nodash}/{primary}"
            try:
                doc_resp = requests.get(archive_url, headers=headers, timeout=30)
                if doc_resp.status_code != 200:
                    continue
                content = doc_resp.text
                docs.append({
                    "title": f"Tesla {form} Filing",
                    "date": fdate,
                    "url": archive_url,
                    "content": content
                })
            except Exception:
                continue

        return docs
    except Exception as e:
        print(f"SEC fetch failed: {e}")
        return []




## MAIN BLOCK MOVED TO END


def setup_langchain_rag_system(documents):
    """
    Step 4: Setup Langchain-based RAG system (from your notebooks)
    This shows how to use Langchain components for a more structured approach
    """
    print("🔗 Setting up Langchain RAG system...")
    
    # Step 1: Convert documents to Langchain Document format
    print("📝 Converting documents to Langchain format...")
    langchain_docs = []
    for doc in documents:
        langchain_doc = Document(
            page_content=doc["content"],
            metadata={
                "title": doc["title"],
                "date": doc["date"],
                "url": doc["url"],
                "source": "tesla_press_release"
            }
        )
        langchain_docs.append(langchain_doc)
    
    # Step 2: Text splitting (from your notebooks)
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    split_docs = text_splitter.split_documents(langchain_docs)
    print(f"Created {len(split_docs)} document chunks")
    
    # Step 3: Setup embeddings (Langchain wrapper)
    print("Setting up HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Step 4: Create vector store
    print("Creating Langchain vector store...")
    from langchain_community.vectorstores import Chroma
    
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory="./langchain_chroma_db"
    )
    
    # Step 5: Setup Ollama LLM (from your notebooks)
    print("Setting up Ollama LLM...")
    llm = Ollama(model="phi3")
    
    # Add LangSmith tracing tags
    if LANGCHAIN_API_KEY != "YOUR_LANGSMITH_API_KEY_HERE":
        llm = llm.with_config({"tags": ["ollama", "phi3", "tesla-rag"]})
    
    # Step 6: Create retrieval chain
    print("🔗 Creating RetrievalQA chain...")
    
    # Custom prompt template (from your notebooks style)
    prompt_template = """You are an expert financial analyst. Use the following context to answer the question about Tesla's business and investment prospects.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear, informative answer based on the context
- Focus on key financial metrics, business strategies, and growth opportunities
- If the context doesn't contain enough information, say so
- Be specific and cite relevant details from the context

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20}
    )
    
    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    print("✅ Langchain RAG system ready!")
    return qa_chain, vectorstore


def create_advanced_langchain_chain(vectorstore, llm):
    """
    Create an advanced Langchain chain with better LangSmith visualization
    This uses the LCEL (LangChain Expression Language) approach from your notebooks
    """
    print("🔗 Creating Advanced Langchain Chain (LCEL)...")
    
    # Create retriever with diversity; we'll dedupe to unique sources later
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 25}
    )
    
    # Advanced prompt template (from your notebooks style)
    template = """You are an expert financial analyst specializing in Tesla and electric vehicle markets. 
    Use the following context from Tesla's official documents to provide a comprehensive investment analysis.

    Context:
    {context}

    Question: {question}

    Instructions:
    - Analyze the provided context thoroughly
    - Focus on key financial metrics, growth drivers, and strategic initiatives
    - Consider both opportunities and risks
    - Provide specific, actionable insights for long-term investors
    - If the context doesn't contain sufficient information, clearly state what's missing

    Investment Analysis:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Helper to format and deduplicate docs by URL/title
    def format_unique_docs(docs, limit=3):
        seen = set()
        unique_docs = []
        for d in docs:
            key = d.metadata.get("url") or d.metadata.get("title") or ""
            if key in seen:
                continue
            seen.add(key)
            unique_docs.append(d)
            if len(unique_docs) >= limit:
                break
        parts = []
        for d in unique_docs:
            title = d.metadata.get("title", "Unknown Title")
            date = d.metadata.get("date", "Unknown Date")
            parts.append(f"Title: {title}\nDate: {date}\n\n{d.page_content}")
        return "\n\n---\n\n".join(parts) if parts else ""

    # Create the chain using LCEL with deduplication
    chain = (
        {"context": retriever | RunnableLambda(lambda ds: format_unique_docs(ds, limit=3)), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Add LangSmith metadata if API key is provided
    if LANGCHAIN_API_KEY != "YOUR_LANGSMITH_API_KEY_HERE":
        chain = chain.with_config({
            "tags": ["tesla-rag", "investment-analysis", "lcel-chain"],
            "metadata": {
                "model": "phi3",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "vector_db": "chroma",
                "retrieval_k": 3
            }
        })
    
    print("Advanced LCEL chain created")
    return chain, retriever


def interactive_chain_mode():
    """
    Interactive mode to test Langchain LCEL chains
    """
    print("Interactive Langchain Chain Testing")
    print("=" * 50)
    
    # Load documents
    with open("tesla_press_releases.json", "r") as f:
        documents = json.load(f)
    
    if not documents:
        print("No documents found. Run the scraper first!")
        return
    
    # Setup Langchain system
    print("Setting up Langchain LCEL chain...")
    qa_chain, vectorstore = setup_langchain_rag_system(documents)
    
    # Create advanced LCEL chain
    llm = Ollama(model="phi3")
    advanced_chain, retriever = create_advanced_langchain_chain(vectorstore, llm)
    
    print("\n Interactive Chain Testing!")
    print("Commands:")
    print("  ask <question> - Test the LCEL chain")
    print("  quit - Exit")
    print(f"View chains at: https://smith.langchain.com")
    
    while True:
        try:
            user_input = input("\nEnter command: ").strip()
            
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            
            if user_input.lower().startswith("ask "):
                question = user_input[4:].strip()
                
                if question:
                    print(f"\nQuestion: {question}")
                    print("=" * 60)
                    print("🔗 Processing with LCEL Chain...")
                    
                    try:
                        # Use the advanced LCEL chain
                        result = advanced_chain.invoke(question)
                        
                        print(f"\Investment Analysis:")
                        print(result)
                        print(f"\nView this chain execution at: https://smith.langchain.com")
                        
                    except Exception as e:
                        print(f"Error: {e}")
                
            else:
                print("Use 'ask <question>' or 'quit'")
                
        except KeyboardInterrupt:
            print("\nGoodbye")
            break
        except Exception as e:
            print(f"Error: {e}")


def interactive_rag_mode():
    """
    Deprecated: previously compared custom RAG vs LangChain. Kept for backward compatibility.
    """
    print("This mode is deprecated. Use interactive_chain_mode() instead.")



# Ensure all functions are defined before calling
if __name__ == "__main__":
    # Step 1: Scrape fresh Tesla press releases
    print("Tesla RAG System - Step by Step")
    print("=" * 50)
    
    press_releases = scrape_press_releases(num_docs=5)
    if not press_releases:
        print("Press releases unavailable. Fetching SEC filings as source documents...")
        press_releases = fetch_tesla_sec_filings(max_docs=5)
    if not press_releases:
        print("No documents retrieved from press or SEC. Exiting.")
        exit(1)

    # Print to console
    print("\nScraped Press Releases:")
    for pr in press_releases:
        print(f"  • {pr['title']} - {pr['date']}")
        print(f"    {pr['content'][:100]}...\n")

    # Save fresh scrape to JSON
    with open("tesla_press_releases.json", "w") as f:
        json.dump(press_releases, f, indent=2)
    print("Saved fresh documents to tesla_press_releases.json")
    
    # Step 2: Interactive LCEL chain testing
    print("\n" + "="*50)
    user_input = input("\nTry interactive LCEL chain testing? (y/n): ").strip().lower()
    if user_input == 'y':
        interactive_chain_mode()
    else:
        print("To try interactive mode later, call: interactive_chain_mode()")