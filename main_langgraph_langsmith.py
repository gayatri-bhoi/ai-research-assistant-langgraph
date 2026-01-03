import streamlit as st
import os
from typing import TypedDict, Annotated
import operator
from dotenv import load_dotenv
import time
import json
from datetime import datetime

# LangGraph & LangChain imports
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. INITIALIZE SESSION STATE ---
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# --- 3. CUSTOM CSS FOR PROFESSIONAL ANIMATIONS ---
st.markdown("""
<style>
    /* Main container animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    /* Header styling */
    .main-header {
        animation: fadeIn 1s ease-out;
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .history-item {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .history-item:hover {
        background: #e9ecef;
        transform: translateX(5px);
    }
    
    .history-time {
        font-size: 0.8rem;
        color: #6c757d;
    }
    
    /* Input container */
    .input-container {
        animation: fadeIn 1.2s ease-out;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.3);
        margin: 2rem 0;
    }
    
    /* Enhanced search input */
    .stTextInput > div > div > input {
        background: white !important;
        border: 2px solid rgba(255,255,255,0.3) !important;
        border-radius: 12px !important;
        padding: 1rem 1.5rem !important;
        font-size: 1.1rem !important;
        color: #2c3e50 !important;
        caret-color: #667eea !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border: 2px solid white !important;
        box-shadow: 0 0 20px rgba(255,255,255,0.5) !important;
        transform: scale(1.02);
        caret-color: #667eea !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #95a5a6 !important;
    }
    
    .stTextInput > label {
        color: white !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Step cards */
    .step-card {
        animation: slideIn 0.5s ease-out;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        color: #2c3e50;
        font-weight: 500;
        font-size: 1.05rem;
    }
    
    .step-card:hover {
        transform: translateX(10px);
    }
    
    /* Pipeline container */
    .pipeline-container {
        background: transparent;
        padding: 1rem 0;
        margin: 1rem 0;
    }
    
    .pipeline-header {
        color: #2c3e50;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Answer container */
    .answer-container {
        animation: fadeIn 1.5s ease-out;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border: 2px solid #667eea;
        margin-top: 2rem;
    }
    
    .answer-container h3 {
        color: #2c3e50;
    }
    
    .answer-container p {
        color: #34495e;
        line-height: 1.8;
    }
    
    /* Loading animation */
    .loading-bar {
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        background-size: 200% 100%;
        animation: shimmer 2s infinite;
        border-radius: 2px;
        margin: 1rem 0;
    }
    
    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        animation: pulse 1s infinite;
    }
    
    /* Status indicators */
    .status-working {
        display: inline-block;
        animation: pulse 1.5s infinite;
    }
    
    /* Emoji animations */
    .emoji-bounce {
        display: inline-block;
        animation: bounce 1s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
</style>
""", unsafe_allow_html=True)

# --- 4. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>⚙️ Controls</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎛️ Actions")
    
    # Clear cache button
    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.cache_resource.clear()
        st.success("✅ Cache cleared successfully!")
        time.sleep(1)
        st.rerun()
    
    # Delete history button
    if st.button("📝 Delete History", use_container_width=True):
        st.session_state.search_history = []
        st.success("✅ History deleted successfully!")
        time.sleep(1)
        st.rerun()
    
    st.markdown("---")
    
    # Search History Section
    st.markdown("### 📚 Search History")
    
    if st.session_state.search_history:
        st.markdown(f"**Total Searches:** {len(st.session_state.search_history)}")
        
        # Display history items
        for idx, item in enumerate(reversed(st.session_state.search_history[-10:])):  # Show last 10
            with st.expander(f"🔍 {item['query'][:40]}...", expanded=False):
                st.markdown(f"**Time:** {item['timestamp']}")
                st.markdown(f"**Method:** {item['method']}")
                if st.button(f"Rerun", key=f"rerun_{idx}"):
                    st.session_state.selected_query = item['query']
                    st.rerun()
    else:
        st.info("No search history yet. Start researching!")
    
    st.markdown("---")
    
    # Statistics
    st.markdown("### 📊 Statistics")
    if st.session_state.search_history:
        search_count = sum(1 for item in st.session_state.search_history if item['method'] == 'Web Search')
        direct_count = len(st.session_state.search_history) - search_count
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🌐 Web Searches", search_count)
        with col2:
            st.metric("💡 Direct Answers", direct_count)
    
    st.markdown("---")
    
    # About Section
    with st.expander("ℹ️ About"):
        st.markdown("""
        **AI Research Assistant v2.0**
        
        Features:
        - Multi-agent architecture
        - Real-time web search
        - LLM-powered synthesis
        - Search history tracking
        - Cache management
        
        Built with LangGraph, Groq & Tavily
        """)

# --- 5. HEADER ---
st.markdown("""
<div class="main-header">
    <h1>🔍 AI Research Assistant</h1>
    <p>Powered by LangGraph & Tavily | Multi-Agent Intelligence</p>
</div>
""", unsafe_allow_html=True)

# --- 6. API KEY CONFIGURATION ---
def load_api_keys():
    """Load API keys from environment variables"""
    groq_key = os.getenv("GROQ_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    langchain_key = os.getenv("LANGCHAIN_API_KEY")
    
    # Strip whitespace and validate
    groq_key = groq_key.strip() if groq_key else None
    tavily_key = tavily_key.strip() if tavily_key else None
    langchain_key = langchain_key.strip() if langchain_key else None
    
    return groq_key, tavily_key, langchain_key

def verify_groq_key(api_key):
    """Test if Groq API key is valid"""
    try:
        test_llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key, timeout=10)
        test_llm.invoke("test")
        return True
    except Exception as e:
        return False

def verify_tavily_key(api_key):
    """Test if Tavily API key is valid"""
    try:
        test_client = TavilyClient(api_key=api_key)
        test_client.search(query="test", max_results=1)
        return True
    except Exception as e:
        return False

groq_key, tavily_key, langchain_key = load_api_keys()

# Verify API keys
groq_valid = False
tavily_valid = False

if groq_key:
    with st.spinner("Verifying Groq API key..."):
        groq_valid = verify_groq_key(groq_key)

if tavily_key:
    with st.spinner("Verifying Tavily API key..."):
        tavily_valid = verify_tavily_key(tavily_key)

# Display configuration status
col1, col2, col3 = st.columns(3)
with col1:
    status = "✅ Valid & Connected" if groq_valid else ("⚠️ Invalid Key" if groq_key else "❌ Key Missing")
    color = "#28a745" if groq_valid else ("#ffc107" if groq_key else "#dc3545")
    st.markdown(f"""
    <div class="info-card" style="border-left-color: {color}">
        <h4>🤖 LLM Status</h4>
        <p>{status}</p>
    </div>
    """, unsafe_allow_html=True)
    
with col2:
    status = "✅ Valid & Connected" if tavily_valid else ("⚠️ Invalid Key" if tavily_key else "❌ Key Missing")
    color = "#28a745" if tavily_valid else ("#ffc107" if tavily_key else "#dc3545")
    st.markdown(f"""
    <div class="info-card" style="border-left-color: {color}">
        <h4>🌐 Search Status</h4>
        <p>{status}</p>
    </div>
    """, unsafe_allow_html=True)
    
with col3:
    status = "✅ Active" if langchain_key else "⚠️ Optional"
    color = "#28a745" if langchain_key else "#6c757d"
    st.markdown(f"""
    <div class="info-card" style="border-left-color: {color}">
        <h4>📊 Tracing Status</h4>
        <p>{status}</p>
    </div>
    """, unsafe_allow_html=True)

# Show detailed error messages
if not groq_key or not tavily_key:
    st.error("⚠️ **Missing API Keys!** Please configure your .env file")
    st.code("""
GROQ_API_KEY=your_actual_groq_key_here
TAVILY_API_KEY=your_actual_tavily_key_here
LANGCHAIN_API_KEY=your_langchain_key_here (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=research-assistance-project
    """)
    st.info("📌 Get your API keys from:\n- Groq: https://console.groq.com/keys\n- Tavily: https://app.tavily.com/")
    st.stop()

if groq_key and not groq_valid:
    st.error("❌ **Invalid Groq API Key!** Please check your GROQ_API_KEY in .env file")
    st.info("🔑 Get a valid API key from: https://console.groq.com/keys")
    st.stop()

if tavily_key and not tavily_valid:
    st.error("❌ **Invalid Tavily API Key!** Please check your TAVILY_API_KEY in .env file")
    st.info("🔑 Get a valid API key from: https://app.tavily.com/")
    st.stop()

# --- 7. CORE LOGIC (Nodes & Graph) ---
class ResearchState(TypedDict):
    query: str
    needs_search: bool
    search_results: str
    final_answer: str
    steps: Annotated[list[str], operator.add]

@st.cache_resource
def get_graph(_groq_api_key, _tavily_api_key):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, groq_api_key=_groq_api_key)
    tavily_client = TavilyClient(api_key=_tavily_api_key)

    def analyze_query(state: ResearchState):
        prompt = f"Analyze if this query needs web search or direct answer: {state['query']}\nRespond 'SEARCH' or 'DIRECT'."
        response = llm.invoke(prompt)
        needs_search = "SEARCH" in response.content.upper()
        return {"needs_search": needs_search, "steps": [f"🧠 Decision: {'Web Search Required' if needs_search else 'Direct Knowledge Available'}"]}

    def search_web(state: ResearchState):
        search_response = tavily_client.search(query=state["query"], max_results=3)
        results = "\n\n".join([f"Source: {r['url']}\n{r['content']}" for r in search_response['results']])
        return {"search_results": results, "steps": [f"🌐 Retrieved {len(search_response['results'])} sources from Tavily"]}

    def synthesize_answer(state: ResearchState):
        prompt = f"Using these results: {state['search_results']}\n\nAnswer the query: {state['query']}"
        response = llm.invoke(prompt)
        return {"final_answer": response.content, "steps": ["✍️ Synthesized comprehensive research report"]}

    def direct_answer(state: ResearchState):
        response = llm.invoke(f"Answer concisely: {state['query']}")
        return {"final_answer": response.content, "steps": ["💡 Generated answer from internal knowledge base"]}

    def route_query(state: ResearchState):
        return "search" if state["needs_search"] else "direct"

    # Build Graph
    workflow = StateGraph(ResearchState)
    workflow.add_node("analyze", analyze_query)
    workflow.add_node("search", search_web)
    workflow.add_node("synthesize", synthesize_answer)
    workflow.add_node("direct", direct_answer)
    
    workflow.set_entry_point("analyze")
    workflow.add_conditional_edges("analyze", route_query, {"search": "search", "direct": "direct"})
    workflow.add_edge("search", "synthesize")
    workflow.add_edge("synthesize", END)
    workflow.add_edge("direct", END)
    
    return workflow.compile()

# --- 8. USER INTERFACE ---
query = st.text_input(
    "🔎 What would you like to research?",
    value=st.session_state.get('selected_query', ''),
    placeholder="e.g., What are the latest AI breakthroughs in 2024?",
    key="search_query"
)

# Clear selected query after use
if st.session_state.get('selected_query'):
    st.session_state.selected_query = ''

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run_button = st.button("🚀 Start Research", use_container_width=True)

if run_button and query:
    try:
        # Initialize graph
        app = get_graph(groq_key, tavily_key)
        
        # Show loading animation
        st.markdown('<div class="loading-bar"></div>', unsafe_allow_html=True)
        
        # Initial State
        initial_input = {
            "query": query,
            "needs_search": False,
            "search_results": "",
            "final_answer": "",
            "steps": []
        }
        
        # Execute with progress updates
        with st.spinner("🤖 Agent analyzing your query..."):
            result = app.invoke(initial_input)
        
        # Display pipeline steps
        st.markdown('<div class="pipeline-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="pipeline-header">🔄 Research Pipeline</h3>', unsafe_allow_html=True)
        
        for i, step in enumerate(result["steps"]):
            time.sleep(0.2)  # Slight delay for visual effect
            st.markdown(f'<div class="step-card">{step}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add to history
        method = "Web Search" if result["needs_search"] else "Direct Answer"
        st.session_state.search_history.append({
            "query": query,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": method
        })
        
        # Display Results
        st.markdown("---")
        st.markdown("### 📝 Research Results")
        st.markdown(result["final_answer"])
        
        # Optional: Show raw sources if search was used
        if result["search_results"]:
            with st.expander("🔍 View Source Data"):
                st.text(result["search_results"])
        
        # Success message
        st.success("✅ Research completed successfully!")
    
    except Exception as e:
        st.error(f"❌ **Error during research:** {str(e)}")
        
        if "401" in str(e) or "Invalid API Key" in str(e):
            st.error("🔑 **API Key Error!** Your API key is invalid or expired.")
            st.info("Please check:\n1. Your .env file has the correct API keys\n2. Keys have no extra spaces or quotes\n3. Keys are active and not expired")
        elif "429" in str(e):
            st.error("⏱️ **Rate Limit!** Too many requests. Please wait a moment.")
        else:
            st.error("Please check your API keys and try again.")
            with st.expander("🐛 View Error Details"):
                st.code(str(e))

elif run_button and not query:
    st.warning("⚠️ Please enter a research query to begin.")

# --- 9. FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Built with ❤️ using LangGraph, Groq & Tavily</p>
    <p style="font-size: 0.9rem;">Multi-Agent Architecture | Real-time Web Search | LLM-Powered Synthesis</p>
</div>
""", unsafe_allow_html=True)