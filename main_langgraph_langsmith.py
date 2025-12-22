import streamlit as st
import os
from typing import TypedDict, Annotated
import operator
from datetime import datetime
from dotenv import load_dotenv
import time

# LangGraph & LangChain imports
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 0;
    }
    
    .stApp {
        background: transparent;
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        animation: fadeIn 0.8s ease-out;
    }
    
    .hero-section h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .hero-section p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.2rem;
        margin-top: 1rem;
        font-weight: 500;
    }
    
    /* Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
    }
    
    /* Metric Cards */
    .metric-container {
        display: flex;
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .metric-box {
        flex: 1;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.95rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Step Timeline */
    .step-timeline {
        position: relative;
        padding: 2rem 0;
    }
    
    .step-item {
        display: flex;
        align-items: center;
        margin: 1rem 0;
        animation: slideInLeft 0.5s ease-out;
    }
    
    .step-icon {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-right: 1rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    .step-analyze { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .step-search { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .step-synthesize { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .step-direct { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
    
    .step-content {
        flex: 1;
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 3px solid #667eea;
        color: white;
        font-weight: 500;
    }
    
    /* Answer Box */
    .answer-box {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 15px 45px rgba(0, 0, 0, 0.3);
        color: white;
        line-height: 1.8;
        font-size: 1.05rem;
    }
    
    /* Source Cards */
    .source-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .source-card:hover {
        transform: translateX(10px);
        border-color: #667eea;
    }
    
    .source-title {
        color: #667eea;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .source-link {
        color: #4facfe;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.3s ease;
    }
    
    .source-link:hover {
        color: #00f2fe;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 12px;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6);
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.08);
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        color: white;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Section Headers */
    .section-header {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        color: white !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="hero-section">
    <h1>🔬 AI Research Assistant</h1>
    <p>Intelligent Multi-Agent System • Powered by LangGraph & Llama 3.3</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Load API keys from .env (hidden)
    groq_key = os.getenv("GROQ_API_KEY", "")
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    
    # Show status only, not the keys
    st.markdown("### 🔐 API Keys Status")
    
    if groq_key:
        st.success("✅ Groq API Key: Loaded from .env")
    else:
        st.error("❌ Groq API Key: Not found")
    
    if tavily_key:
        st.success("✅ Tavily API Key: Loaded from .env")
    else:
        st.error("❌ Tavily API Key: Not found")
    
    st.info("💡 Add your API keys to the `.env` file in your project root")
    
    st.divider()
    
    # Model Configuration
    st.markdown("### 🤖 Model Settings")
    
    model = st.selectbox(
        "Select Model",
        ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
        help="Choose the language model"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Creativity level: Lower = Focused, Higher = Creative"
    )
    
    max_results = st.slider(
        "Search Results",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of sources to retrieve"
    )
    
    st.divider()
    
    # Status
    if groq_key and tavily_key:
        st.success("✅ Ready to Research")
    else:
        st.error("❌ API Keys Required")
        st.info("💡 Add keys to .env file or enter above")
    
    st.divider()
    
    st.markdown("### 📚 Features")
    st.markdown("""
    - 🧠 **Smart Routing**: Auto-decides between knowledge & search
    - 🌐 **Web Search**: Real-time information via Tavily
    - 🔄 **Multi-Step**: LangGraph orchestration
    - ⚡ **Fast**: Optimized Llama 3.3 inference
    """)
    
    st.divider()
    
    st.markdown("### 🎯 How It Works")
    st.markdown("""
    1. **Analyze**: Determines if search is needed
    2. **Search**: Queries web if required
    3. **Synthesize**: Combines sources
    4. **Answer**: Delivers comprehensive response
    """)

# --- STATE DEFINITION ---
class ResearchState(TypedDict):
    query: str
    needs_search: bool
    search_results: str
    final_answer: str
    steps: Annotated[list[str], operator.add]
    sources: list[dict]

# --- GRAPH BUILDER ---
def get_graph(groq_api_key, tavily_api_key, model_name, temp, max_res):
    llm = ChatGroq(model=model_name, temperature=temp, groq_api_key=groq_api_key)
    tavily_client = TavilyClient(api_key=tavily_api_key)

    def analyze_query(state: ResearchState):
        prompt = f"""Analyze whether this query requires real-time web search or can be answered from existing knowledge.

Query: "{state['query']}"

Consider:
- Does it ask about recent events, current news, or today's data?
- Does it need real-time information?
- Could facts have changed since training cutoff?

Respond with ONLY 'SEARCH' or 'DIRECT'."""
        
        response = llm.invoke(prompt)
        needs_search = "SEARCH" in response.content.upper()
        
        return {
            "needs_search": needs_search,
            "steps": [f"🧠|analyze|{'Web search required for current data' if needs_search else 'Answering from knowledge base'}"],
            "sources": []
        }

    def search_web(state: ResearchState):
        search_response = tavily_client.search(query=state["query"], max_results=max_res)
        
        results = "\n\n".join([
            f"Source {i+1}:\nTitle: {r.get('title', 'Untitled')}\nURL: {r['url']}\nContent: {r['content'][:500]}..."
            for i, r in enumerate(search_response['results'])
        ])
        
        sources = [
            {"url": r['url'], "title": r.get('title', 'Untitled Source')} 
            for r in search_response['results']
        ]
        
        return {
            "search_results": results,
            "steps": [f"🌐|search|Found {len(search_response['results'])} relevant sources"],
            "sources": sources
        }

    def synthesize_answer(state: ResearchState):
        prompt = f"""Based on these web search results, provide a comprehensive and well-structured answer.

Search Results:
{state['search_results']}

Original Query: {state['query']}

Instructions:
- Synthesize information from multiple sources
- Provide specific facts and data
- Be comprehensive yet clear
- Cite key findings naturally"""
        
        response = llm.invoke(prompt)
        return {
            "final_answer": response.content,
            "steps": ["✍️|synthesize|Created comprehensive answer from sources"]
        }

    def direct_answer(state: ResearchState):
        prompt = f"""Provide a detailed, accurate, and well-structured answer to this query:

{state['query']}

Be comprehensive and informative."""
        
        response = llm.invoke(prompt)
        return {
            "final_answer": response.content,
            "steps": ["💡|direct|Generated answer from knowledge base"]
        }

    def route_query(state: ResearchState):
        return "search" if state["needs_search"] else "direct"

    # Build Workflow Graph
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

# --- MAIN INTERFACE ---
st.markdown('<div class="section-header">🔍 Research Query</div>', unsafe_allow_html=True)

query = st.text_input(
    "What would you like to research?",
    placeholder="e.g., What are the latest developments in quantum computing?",
    label_visibility="collapsed",
    key="main_query"
)

# Example Queries
with st.expander("💡 Try These Example Queries"):
    col1, col2, col3 = st.columns(3)
    
    examples = [
        ("🤖 AI Breakthroughs", "What are the latest AI breakthroughs in December 2024?"),
        ("📈 Market Trends", "What are current global stock market trends?"),
        ("🌍 Climate Updates", "What are recent developments in climate change research?"),
        ("🚀 Space News", "What are the latest space exploration achievements?"),
        ("💊 Medical Research", "What are recent breakthroughs in cancer research?"),
        ("⚡ Tech Innovations", "What are the latest technological innovations?")
    ]
    
    for i, (label, example_query) in enumerate(examples):
        col = [col1, col2, col3][i % 3]
        with col:
            if st.button(label, use_container_width=True, key=f"example_{i}"):
                query = example_query
                st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

run_button = st.button("🚀 Start Research", use_container_width=True, type="primary")

# --- EXECUTION LOGIC ---
if run_button and query and groq_key and tavily_key:
    
    with st.spinner("🔄 Initializing AI Research System..."):
        time.sleep(0.5)  # Brief pause for effect
        app = get_graph(groq_key, tavily_key, model, temperature, max_results)
        
        initial_input = {
            "query": query,
            "needs_search": False,
            "search_results": "",
            "final_answer": "",
            "steps": [],
            "sources": []
        }
        
        start_time = datetime.now()
        result = app.invoke(initial_input)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
    
    # Metrics Display
    st.markdown('<div class="section-header">📊 Research Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{len(result['steps'])}</div>
            <div class="metric-label">Steps Executed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{len(result.get('sources', []))}</div>
            <div class="metric-label">Sources Found</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{duration:.2f}s</div>
            <div class="metric-label">Processing Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Execution Steps
    st.markdown('<div class="section-header">🔄 Execution Timeline</div>', unsafe_allow_html=True)
    
    for step in result["steps"]:
        if "|" in step:
            emoji, step_type, message = step.split("|", 2)
            st.markdown(f"""
            <div class="step-item">
                <div class="step-icon step-{step_type}">{emoji}</div>
                <div class="step-content">{message}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Final Answer
    st.markdown('<div class="section-header">📝 Research Results</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-box">{result["final_answer"]}</div>', unsafe_allow_html=True)
    
    # Sources
    if result.get("sources"):
        st.markdown('<div class="section-header">🔗 Sources</div>', unsafe_allow_html=True)
        
        for idx, source in enumerate(result["sources"], 1):
            st.markdown(f"""
            <div class="source-card">
                <div class="source-title">📄 {idx}. {source.get('title', 'Untitled')}</div>
                <a href="{source['url']}" target="_blank" class="source-link">
                    🔗 {source['url'][:60]}...
                </a>
            </div>
            """, unsafe_allow_html=True)
    
    # Raw Search Data
    if result["search_results"]:
        with st.expander("🔍 View Raw Search Data"):
            st.code(result["search_results"], language="text")

elif run_button and not query:
    st.warning("⚠️ Please enter a research query above.")

elif run_button and (not groq_key or not tavily_key):
    st.error("🚫 Please provide API keys in the sidebar or .env file.")

# --- FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: rgba(255, 255, 255, 0.5); padding: 2rem 0; border-top: 1px solid rgba(255, 255, 255, 0.1);">
    <p style="font-size: 1rem; margin-bottom: 0.5rem;">Built with ❤️ using Streamlit, LangGraph & Llama 3.3</p>
    <p style="font-size: 0.9rem;">⚡ Intelligent Routing • 🌐 Real-time Search • 🤖 Advanced AI Reasoning</p>
</div>
""", unsafe_allow_html=True)