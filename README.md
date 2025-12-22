"""
📦 QUICK SETUP GUIDE:

1. Create Virtual Environment:
   python -m venv venv
   
2. Activate Virtual Environment:
   - Windows: venv\Scripts\activate
   - Mac/Linux: source venv/bin/activate

3. Install Dependencies:
   pip install -r requirements.txt

4. Setup Environment Variables:
   - Copy .env.example to .env
   - Add your actual API keys to .env
   
   cp .env.example .env
   
5. Get API Keys:
   - Groq: https://console.groq.com/keys
   - Tavily: https://app.tavily.com/

6. Run the Application:
   streamlit run main_langgraph_langsmith.py

7. Access the App:
   Open browser to: http://localhost:8501

🔐 SECURITY NOTES:
- Never commit .env file to Git
- Keep API keys secret
- Use .env.example for team sharing
- Add .env to .gitignore (already included)

📝 PROJECT STRUCTURE:
your-project/
├── main_langgraph_langsmith.py  # Main application
├── requirements.txt              # Python dependencies
├── .env                          # Your API keys (not committed)
├── .env.example                  # Template for API keys
├── .gitignore                    # Git ignore rules
└── README.md                     # Project documentation
