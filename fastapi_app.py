from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import json
import os
import traceback

# Import from main.py with error handling
try:
    from main import setup_langchain_rag_system, create_langchain_chain, fetch_tesla_sec_filings
    print("Successfully imported functions from main.py")
except ImportError as e:
    print(f"Failed to import from main.py: {e}")
    print("Make sure main.py is in the same directory and has no syntax errors")
    raise e

app = FastAPI(title="Tesla RAG Investment Analysis API", version="1.0.0")

# Global variables to store the chain (initialize once)
chain = None
is_initialized = False

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    status: str

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    global chain, is_initialized
    
    if not is_initialized:
        try:
            print("Initializing RAG system...")
            
            # Check if we have documents
            if not os.path.exists("tesla_press_releases.json"):
                print("No documents found. Fetching SEC filings...")
                documents = fetch_tesla_sec_filings(max_docs=20)
                if not documents:
                    print("Failed to fetch documents from SEC")
                    return
                
                # Save documents
                with open("tesla_press_releases.json", "w") as f:
                    json.dump(documents, f, indent=2)
                print(f"Saved {len(documents)} documents")
            else:
                print("Loading existing documents...")
            
            # Load documents
            with open("tesla_press_releases.json", "r") as f:
                documents = json.load(f)
            print(f"Loaded {len(documents)} documents")
            
            # Setup RAG system
            print("Setting up RAG infrastructure...")
            vectorstore, llm = setup_langchain_rag_system(documents)
            
            print("Creating Langchain chain...")
            chain = create_langchain_chain(vectorstore, llm)
            
            is_initialized = True
            print("RAG system ready!")
            
        except Exception as e:
            print(f"Error initializing RAG system: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            is_initialized = False

@app.get("/", response_class=HTMLResponse)
async def root():
    """Simple web interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tesla RAG Investment Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
            input[type="text"] { width: 70%; padding: 10px; margin: 10px 0; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .response { background: white; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Tesla Investment Analysis</h1>
            <p>Ask questions about Tesla's business and get AI-powered investment insights based on official SEC filings.</p>
            
            <div>
                <input type="text" id="question" placeholder="Ask about Tesla's financial performance, growth strategy, risks..." />
                <button onclick="askQuestion()">Ask Question</button>
            </div>
            
            <div id="response"></div>
        </div>

        <script>
            async function askQuestion() {
                const question = document.getElementById('question').value;
                const responseDiv = document.getElementById('response');
                
                if (!question.trim()) {
                    alert('Please enter a question');
                    return;
                }
                
                responseDiv.innerHTML = '<p>Analyzing your question...</p>';
                
                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: question })
                    });
                    
                    const data = await response.json();
                    
                    if (data.status === 'success') {
                        const formattedAnswer = data.answer.split('\\n').join('<br>');
                        responseDiv.innerHTML = `
                            <div class="response">
                                <h3>Investment Analysis:</h3>
                                <p>${formattedAnswer}</p>
                            </div>
                        `;
                    } else {
                        responseDiv.innerHTML = `<p>Error: ${data.answer}</p>`;
                    }
                } catch (error) {
                    responseDiv.innerHTML = `<p>Error: ${error.message}</p>`;
                }
            }
            
            // Allow Enter key to submit
            document.getElementById('question').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    askQuestion();
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system"""
    global chain, is_initialized
    
    print(f"Received query: {request.question}")
    
    if not is_initialized or chain is None:
        print("RAG system not initialized")
        raise HTTPException(status_code=503, detail="RAG system not initialized. Please wait for startup to complete.")
    
    try:
        print("Processing query with RAG chain...")
        # Get response from the chain
        result = chain.invoke(request.question)
        print("Query processed successfully")
        
        return QueryResponse(
            answer=result,
            status="success"
        )
    except Exception as e:
        print(f"Error processing query: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if is_initialized else "initializing",
        "rag_system_ready": is_initialized,
        "chain_available": chain is not None
    }

@app.get("/status")
async def get_status():
    """Detailed status endpoint"""
    return {
        "initialized": is_initialized,
        "chain_ready": chain is not None,
        "documents_exist": os.path.exists("tesla_press_releases.json"),
        "status": "ready" if (is_initialized and chain is not None) else "not_ready"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
