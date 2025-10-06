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
# Yahoo Finance press releases for TSLA
YF_BASE_URL = "https://finance.yahoo.com"
YF_PRESS_URL = "https://finance.yahoo.com/quote/TSLA/press-releases?p=TSLA"

# LangSmith Configuration (for browser visualization)
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = "tesla-rag-system"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
#os.getenv("FMP_API_KEY")

# Set up LangSmith tracing (only if API key is provided)
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = LANGCHAIN_ENDPOINT
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    print("🔗 LangSmith tracing enabled - view your chains at https://smith.langchain.com")
else:
    print("LangSmith API key not set - chains won't be traced to browser")

def scrape_press_releases(num_docs=5):
    """
    Scrape TSLA press releases from Yahoo Finance (server-rendered list).
    Returns [] on failure.
    """
    try:
        from urllib.parse import urljoin
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        print("Scraping Yahoo Finance press releases...")
        # Try cloudscraper first (handles anti-bot); fallback to requests
        list_resp = None
        try:
            import cloudscraper  # optional dependency
            scraper = cloudscraper.create_scraper()
            list_resp = scraper.get(YF_PRESS_URL, headers=headers, timeout=20, allow_redirects=True)
        except Exception:
            list_resp = requests.get(YF_PRESS_URL, headers=headers, timeout=20, allow_redirects=True)
        if list_resp.status_code != 200:
            print(f"Yahoo list fetch failed: {list_resp.status_code}")
            return []
        list_soup = BeautifulSoup(list_resp.text, "html.parser")

        # Yahoo list items link to /news/...
        article_links = []
        for a in list_soup.select('a[href^="/news/"]'):
            href = a.get("href")
            if href:
                article_links.append(urljoin(YF_BASE_URL, href))

        article_links = list(dict.fromkeys(article_links))[: max(num_docs * 4, num_docs)]
        if not article_links:
            print("No Yahoo Finance press links found. Writing yf_debug.html for inspection and trying RSS fallback...")
            try:
                with open("yf_debug.html", "w") as dbg:
                    dbg.write(list_resp.text if list_resp is not None else "")
            except Exception:
                pass
            # RSS fallback (headlines feed)
            try:
                rss_url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TSLA&region=US&lang=en-US"
                rss_resp = requests.get(rss_url, headers=headers, timeout=20)
                if rss_resp.status_code == 200:
                    rss = BeautifulSoup(rss_resp.text, "xml")
                    for item in rss.find_all("item"):
                        link_tag = item.find("link")
                        if link_tag and link_tag.text:
                            article_links.append(link_tag.text.strip())
                    article_links = list(dict.fromkeys(article_links))[: max(num_docs * 4, num_docs)]
            except Exception:
                pass
            if not article_links:
                return []

    docs = []
        for url in article_links:
            if len(docs) >= num_docs:
                break
            try:
                # Use same approach as list: cloudscraper if available
                art_resp = None
                try:
                    import cloudscraper
                    scraper = cloudscraper.create_scraper()
                    art_resp = scraper.get(url, headers=headers, timeout=20, allow_redirects=True)
                except Exception:
                    art_resp = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
                if art_resp.status_code != 200:
                    continue
                art_soup = BeautifulSoup(art_resp.text, "html.parser")

                title_el = art_soup.select_one("h1")
        title = title_el.get_text(strip=True) if title_el else "Untitled"

                date_el = art_soup.select_one("time, .caas-attr-time-style")
        date = date_el.get_text(strip=True) if date_el else "Unknown"

                # Yahoo article body usually under .caas-body
                ps = art_soup.select("div.caas-body p")
                if not ps:
                    ps = art_soup.select("article p")
                content = " ".join([p.get_text(strip=True) for p in ps])
                if len(content) < 200:
                    continue

        docs.append({
            "title": title,
            "date": date,
            "url": url,
            "content": content
        })
            except Exception:
                continue

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

        wanted = {"10-K", "10-Q", "8-K", "DEF 14A", "SC 13G", "SC 13D", "4", "8-A"}
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


def fetch_fmp_press_releases(symbol="TSLA", limit=5):
    """
    Fetch press releases from FinancialModelingPrep.
    Requires FMP_API_KEY in environment. Returns [] if not available or on failure.
    """
    try:
        if not FMP_API_KEY:
            print("FMP_API_KEY not set; skipping FMP press releases.")
            return []
        url = f"https://financialmodelingprep.com/api/v3/press-releases/{symbol}"
        params = {"apikey": FMP_API_KEY}
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code != 200:
            print(f"FMP press releases fetch failed: {resp.status_code}")
            try:
                # Log a small snippet of the response body for diagnostics
                snippet = (resp.text or "")[:200]
                if snippet:
                    print(f"FMP response snippet: {snippet}")
            except Exception:
                pass
            # Fallback to stock_news endpoint for broader news if press-releases blocked
            news_url = "https://financialmodelingprep.com/api/v3/stock_news"
            news_params = {"tickers": symbol, "limit": max(limit, 10), "apikey": FMP_API_KEY}
            news_resp = requests.get(news_url, params=news_params, timeout=20)
            if news_resp.status_code != 200:
                print(f"FMP stock_news fetch failed: {news_resp.status_code}")
                return []
            news = news_resp.json() or []
            docs_news = []
            for item in news[:limit * 2]:
                title = item.get("title") or "Untitled"
                date = item.get("publishedDate") or item.get("date") or "Unknown"
                content = item.get("text") or item.get("content") or ""
                url_item = item.get("url") or ""
                if not content or len(content) < 200:
                    continue
                docs_news.append({
                    "title": title,
                    "date": date,
                    "url": url_item,
                    "content": content
                })
            return docs_news[:limit]
        data = resp.json() or []
        docs = []
        for item in data[:limit]:
            title = item.get("title") or "Untitled"
            date = item.get("date") or item.get("publishedDate") or "Unknown"
            content = item.get("text") or item.get("content") or ""
            url_item = item.get("url") or ""
            # Skip too-short content
            if not content or len(content) < 200:
                continue
            docs.append({
                "title": title,
                "date": date,
                "url": url_item,
                "content": content
            })
        return docs
    except Exception as e:
        print(f"FMP fetch failed: {e}")
        return []


def fetch_fmp_articles(symbol="TSLA", limit=5, pages=1):
    """
    Fetch articles from FMP stable articles endpoint and filter for the given symbol.
    Falls back gracefully if unauthorized. Content HTML is stripped to text.
    """
    try:
        if not FMP_API_KEY:
            print("FMP_API_KEY not set; skipping FMP articles.")
            return []
        base_url = "https://financialmodelingprep.com/stable/fmp-articles"
        headers = {"Accept": "application/json"}
        collected = []
        from bs4 import BeautifulSoup as _BS
        for page in range(max(pages, 1)):
            params = {"page": page, "limit": max(limit * 2, limit), "apikey": FMP_API_KEY}
            resp = requests.get(base_url, params=params, headers=headers, timeout=20)
            if resp.status_code != 200:
                print(f"FMP articles fetch failed (page {page}): {resp.status_code}")
                try:
                    snippet = (resp.text or "")[:200]
                    if snippet:
                        print(f"FMP articles response snippet: {snippet}")
                except Exception:
                    pass
                break
            items = resp.json() or []
            for it in items:
                tickers = (it.get("tickers") or "").upper()
                title = it.get("title") or ""
                content_html = it.get("content") or ""
                link = it.get("link") or ""
                date = it.get("date") or it.get("publishedDate") or "Unknown"
                # Filter for Tesla by ticker or mention
                if ("TSLA" not in tickers) and ("TESLA" not in title.upper()) and ("TESLA" not in content_html.upper()):
                    continue
                text = _BS(content_html, "html.parser").get_text(" ", strip=True)
                if len(text) < 200:
                    continue
                collected.append({
                    "title": title or "Untitled",
                    "date": date,
                    "url": link,
                    "content": text
                })
                if len(collected) >= limit:
                    break
            if len(collected) >= limit:
                break
        return collected
    except Exception as e:
        print(f"FMP articles fetch failed: {e}")
        return []


## MAIN BLOCK MOVED TO END


def setup_langchain_rag_system(documents):
    """
    Setup the complete RAG infrastructure: documents → chunks → embeddings → vector store
    Returns vectorstore and llm for use with create_advanced_langchain_chain()
    """
    print("Setting up Langchain RAG system...")
    
    # Step 1: Convert documents to Langchain Document format
    print("Converting documents to Langchain format...")
    langchain_docs = []
    for doc in documents:
        langchain_doc = Document(
            page_content=doc["content"],
            metadata={
                "title": doc["title"],
                "date": doc["date"],
                "url": doc["url"],
                "source": "tesla_document"
            }
        )
        langchain_docs.append(langchain_doc)
    
    # Step 2: Text splitting
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
    
    # Step 5: Setup Ollama LLM
    print("Setting up Ollama LLM...")
    llm = Ollama(model="phi3")
    
    # Add LangSmith tracing tags
    if LANGCHAIN_API_KEY:
        llm = llm.with_config({"tags": ["ollama", "phi3", "tesla-rag"]})
    
    print("RAG infrastructure ready")
    return vectorstore, llm


def create_langchain_chain(vectorstore, llm):
    """
    Create a Langchain LCEL chain that retrieves the best 3 document matches
    """
    print("Creating Langchain Chain (LCEL)...")
    
    # Create retriever for the best 3 matches
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance for diversity
        search_kwargs={"k": 3, "fetch_k": 20}  # Get best 3 from 10 candidates
    )
    
    # Prompt template for investment analysis
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
    
    # Helper to format the 3 retrieved documents
    def format_docs(docs):
        parts = []
        for i, d in enumerate(docs, 1):
            title = d.metadata.get("title", "Unknown Title")
            date = d.metadata.get("date", "Unknown Date")
            parts.append(f"Document {i}:\nTitle: {title}\nDate: {date}\n\n{d.page_content}")
        return "\n\n---\n\n".join(parts) if parts else ""

    # Create the LCEL chain
    chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Add LangSmith metadata if API key is provided
    if LANGCHAIN_API_KEY:
        chain = chain.with_config({
            "tags": ["tesla-rag", "investment-analysis", "lcel-chain"],
            "metadata": {
                "model": "phi3",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "vector_db": "chroma",
                "retrieval_k": 3
            }
        })
    
    print("LCEL chain created - retrieves best 3 matches")
    return chain


def interactive_chain_mode():
    """
    Interactive mode to test the simplified Langchain chain
    """
    print("Interactive Langchain Chain Testing")
    print("=" * 50)
    
    # Load documents
    with open("tesla_press_releases.json", "r") as f:
        documents = json.load(f)
    
    if not documents:
        print("No documents found. Run the scraper first!")
        return
    
    # Setup RAG infrastructure
    print("Setting up RAG infrastructure...")
    vectorstore, llm = setup_langchain_rag_system(documents)
    
    # Create the single LCEL chain
    print("Creating Langchain chain...")
    chain = create_langchain_chain(vectorstore, llm)
    
    print("Commands:")
    print("  ask <question> - Test the chain (retrieves best 3 documents)")
    print("  quit - Exit")
    if LANGCHAIN_API_KEY:
        print(f"View chains at: https://smith.langchain.com")
    
    while True:
        try:
            user_input = input("\nEnter command: ").strip()
            
            if user_input.lower() == "quit":
                print("Goodbye")
                break
            
            if user_input.lower().startswith("ask "):
                question = user_input[4:].strip()
                
                if question:
                    print(f"\nQuestion: {question}")
                    print("=" * 60)
                    print("Processing with LCEL Chain...")
                    
                    try:
                        # Use the single LCEL chain
                        result = chain.invoke(question)
                        
                        print(f"\nInvestment Analysis:")
                        print(result)
                        
                    except Exception as e:
                        print(f"Error: {e}")
                
            else:
                print("Use 'ask <question>' or 'quit'")
                
        except KeyboardInterrupt:
            print("\nGoodbye")
            break
        except Exception as e:
            print(f"Error: {e}")





# Ensure all functions are defined before calling
if __name__ == "__main__":
    # Step 1: Scrape fresh Tesla press releases
    print("Tesla RAG System - Step by Step")
    print("=" * 50)
    
    # Skip press releases for now - go directly to SEC filings
    print("Fetching SEC filings as source documents...")
    press_releases = fetch_tesla_sec_filings(max_docs=20)
    if not press_releases:
        print("No documents retrieved from SEC. Exiting.")
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