import streamlit as st
import os
import yfinance as yf
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Pre-Market AI Trader Agent", layout="wide", page_icon="📊")
st.title("📈 Pre-Market AI Analyst Dashboard")
st.caption("AI-powered market structure, volume profile, and risk target analysis.")

# Securely input API Key in sidebar or environment
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))


# Configuration inputs
watchlist_input = st.sidebar.text_input("Your Watchlist Tickers (comma separated)", "AAPL, NVDA, TSLA")
trading_style = st.sidebar.selectbox("Trading Style", ["Day Trader", "Swing Trader", "Options Seller"])
risk_pct = st.sidebar.slider("Max Account Risk Per Trade (%)", 0.5, 5.0, 1.0, 0.5)

# --- LANGCHAIN AGENT IMPLEMENTATION ---
if openai_api_key:
    # 1. Define Tools
    @tool
    def get_market_data(ticker: str) -> str:
        """Fetches the last 5 days of daily Open, High, Low, Close, and Volume data for a stock ticker."""
        try:
            stock = yf.Ticker(ticker.strip().upper())
            hist = stock.history(period="5d")
            return f"--- {ticker.upper()} Market Data ---\n{hist.to_string()}"
        except Exception as e:
            return f"Error fetching data for {ticker}: {str(e)}"

    @tool
    def get_ticker_news(ticker: str) -> str:
        """Fetches recent headlines and sentiment catalysts for a specific ticker."""
        try:
            stock = yf.Ticker(ticker.strip().upper())
            news_items = stock.news[:3]
            return f"--- {ticker.upper()} Recent News ---\n" + "\n".join([f"- {n['title']}" for n in news_items])
        except Exception as e:
            return f"Error fetching news for {ticker}: {str(e)}"

    tools = [get_market_data, get_ticker_news]

    # 2. Setup LLM and Prompt Engineering
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1, openai_api_key=openai_api_key, base_url=os.environ.get("OPENAI_API_URL", "https://aibe.mygreatlearning.com/openai/v1"))

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an elite quantitative analyst and trading coach. 
         Your job is to analyze the tickers requested using your tools.
         User Context: Style: {trading_style}, Max Risk: {risk_pct}%.
         
         Output Structure Required:
         ### 📊 Macro & Volume Pulse
         - Summarize overall volume and price direction.
         
         ### 📌 Ticker Breakdowns
         - For each asset, define Market Structure (Bullish/Bearish/Range), Key Volume Nodes, and Support/Resistance lines.
         
         ### 🎯 Actionable Alert Setups (Day/Week)
         - Provide exact Entry Trigger, Stop Loss, and Targets (Min 1:2 Risk/Reward).
         """),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # 3. Assemble Agent
    agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=prompt)

    # --- UI INTERACTION ---
    if st.button("🚀 Run Pre-Market Analysis", type="primary"):
        with st.spinner("Agent parsing market structures and analyzing order data..."):
            try:
                # Trigger the execution loop
                query = f"Provide today's morning preparation report for these assets: {watchlist_input}"
                response = agent.invoke({"input": query})

                # Render markdown output gracefully in UI
                st.markdown("---")
                st.markdown(response["output"])
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
else:
    st.warning("⚠️ Please provide an OpenAI API Key in the left sidebar to start the agent.")
