import streamlit as st
import requests
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .step-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1E88E5;
    }
    .answer-box {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        border-left: 4px solid #4CAF50;
    }
    .query-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #FF9800;
    }
    </style>
""", unsafe_allow_html=True)

# API endpoint
API_URL = "http://localhost:8000"

# Header
st.markdown('<div class="main-header">🔍 AI Research Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by LangGraph + Groq + Tavily</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This AI Research Assistant uses:
    - **LangGraph** for workflow orchestration
    - **Groq** for fast LLM inference
    - **Tavily** for web search
    - **FastAPI** as backend
    - **Streamlit** for UI
    """)
    
    st.divider()
    
    st.header("🎯 How It Works")
    st.write("""
    1. Enter your question
    2. AI analyzes if web search is needed
    3. Searches web OR uses knowledge base
    4. Generates comprehensive answer
    """)
    
    st.divider()
    
    # Check API health
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ Backend Connected")
        else:
            st.error("⚠️ Backend Issue")
    except:
        st.error("❌ Backend Offline")
        st.info("Run: `uvicorn backend:app --reload`")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💭 Ask a Question")
    
    # Initialize selected example in session state
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = ""
    
    # Example questions (placed BEFORE the text area)
    st.caption("**Try these examples:**")
    example_col1, example_col2 = st.columns(2)
    
    with example_col1:
        if st.button("🌍 Latest AI news"):
            st.session_state.selected_example = "What are the latest developments in AI?"
        if st.button("📊 Stock market today"):
            st.session_state.selected_example = "What happened in the stock market today?"
    
    with example_col2:
        if st.button("🏛️ Capital of France"):
            st.session_state.selected_example = "What is the capital of France?"
        if st.button("🔬 Quantum computing"):
            st.session_state.selected_example = "Explain quantum computing in simple terms"
    
    # Query input (uses selected example as default value)
    user_query = st.text_area(
        "Enter your question:",
        value=st.session_state.selected_example,
        placeholder="e.g., What are the latest AI developments? or What is the capital of France?",
        height=100,
        key="query_input"
    )
    
    # Clear the selected example after it's been used
    if st.session_state.selected_example:
        st.session_state.selected_example = ""
    
    # Submit button
    submit_button = st.button("🚀 Get Answer", type="primary", use_container_width=True)

with col2:
    st.header("📊 Stats")
    
    # Initialize session state for history
    if 'query_count' not in st.session_state:
        st.session_state.query_count = 0
    if 'search_count' not in st.session_state:
        st.session_state.search_count = 0
    
    st.metric("Total Queries", st.session_state.query_count)
    st.metric("Web Searches", st.session_state.search_count)

# Process query
if submit_button and user_query:
    with st.spinner("🤔 Processing your query..."):
        try:
            # Call FastAPI backend
            response = requests.post(
                f"{API_URL}/ask",
                json={"query": user_query},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Update stats
                st.session_state.query_count += 1
                if result["needs_search"]:
                    st.session_state.search_count += 1
                
                # Display query
                st.markdown("---")
                st.markdown(f'<div class="query-box"><strong>📝 Your Question:</strong><br>{result["query"]}</div>', unsafe_allow_html=True)
                
                # Display steps
                st.subheader("🔄 Processing Steps")
                for step in result["steps"]:
                    st.markdown(f'<div class="step-box">{step}</div>', unsafe_allow_html=True)
                
                # Display answer
                st.subheader("💡 Answer")
                st.markdown(f'<div class="answer-box">{result["final_answer"]}</div>', unsafe_allow_html=True)
                
                # Add to history
                if 'history' not in st.session_state:
                    st.session_state.history = []
                
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "query": result["query"],
                    "answer": result["final_answer"],
                    "used_search": result["needs_search"]
                })
                
                # Download button
                st.download_button(
                    label="📥 Download Answer",
                    data=f"Query: {result['query']}\n\nAnswer: {result['final_answer']}",
                    file_name=f"answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            else:
                st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend. Make sure it's running!")
            st.code("uvicorn backend:app --reload")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

elif submit_button:
    st.warning("⚠️ Please enter a question first!")

# Query History
if 'history' in st.session_state and st.session_state.history:
    st.markdown("---")
    st.header("📜 Query History")
    
    for idx, item in enumerate(reversed(st.session_state.history[-5:])):  # Show last 5
        with st.expander(f"🕐 {item['timestamp']} - {item['query'][:50]}..."):
            st.write(f"**Query:** {item['query']}")
            st.write(f"**Used Web Search:** {'Yes ✓' if item['used_search'] else 'No ✗'}")
            st.write(f"**Answer:** {item['answer'][:200]}...")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    Built with ❤️ using LangGraph, FastAPI & Streamlit | 
    <a href='https://github.com/langchain-ai/langgraph' target='_blank'>LangGraph Docs</a>
</div>
""", unsafe_allow_html=True)