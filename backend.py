from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Import our graph logic
from graph_logic import create_graph

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="AI Research Assistant API",
    description="LangGraph-powered research assistant with web search",
    version="1.0.0"
)

# Add CORS middleware (allows Streamlit to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create graph instance
graph = create_graph()

# Request/Response models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    final_answer: str
    steps: list[str]
    needs_search: bool

# Routes
@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "AI Research Assistant API is running!",
        "endpoints": {
            "/ask": "POST - Ask a question",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
def health_check():
    """Check if all services are configured"""
    missing = []
    if not os.getenv("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY")
    if not os.getenv("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")
    
    if missing:
        return {
            "status": "warning",
            "message": f"Missing API keys: {', '.join(missing)}"
        }
    
    return {
        "status": "healthy",
        "message": "All services configured"
    }

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    """
    Process a research query using LangGraph
    
    - **query**: The question to research
    """
    try:
        # Run the graph
        result = graph.invoke({
            "query": request.query,
            "needs_search": False,
            "search_results": "",
            "final_answer": "",
            "steps": []
        })
        
        return QueryResponse(
            query=request.query,
            final_answer=result["final_answer"],
            steps=result["steps"],
            needs_search=result["needs_search"]
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

# Run with: uvicorn backend:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)