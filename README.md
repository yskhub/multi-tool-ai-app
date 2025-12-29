# multi-tool-ai-app
# 🧰 Multi-Tool AI App (Streamlit + Groq)

A single Streamlit web app with 3 AI tools:
1) **Text Summarizer** . Paste long text → get a short summary  
2) **Idea Generator** . Enter a topic → get multiple ideas  
3) **Simple Chatbot** . Ask questions → get AI answers  

This project uses **Python + Streamlit + Groq API** and is designed to be deployed on **Streamlit Community Cloud**.

---

## ✅ Features

- **Manual API Key Input (BYOK)**: User pastes their Groq API key in the sidebar (masked password field)
- **Test API Key Button**: One click to confirm the key works
- **3 tools in one app** using Streamlit tabs
- **Model selector**: choose a smaller fast model or a larger higher-quality model
- **Chat history** stored in session state (with a clear chat button)

---

## 🧱 Tech Stack

- Python
- Streamlit (UI)
- Groq Python SDK (LLM API)

---

## 🔑 Groq API Key (How it works)

This app does **not** use `.streamlit/secrets.toml`.

Instead, the user enters the API key manually in the app:
- Sidebar → paste key → click **✅ Test API Key**
- Key is stored only in the current Streamlit session (`st.session_state`)

> Note: In a deployed Streamlit app, the key is sent to the server session to make API calls.
> This is normal for “Bring Your Own Key” demos.

---

## README Summary

### 1) What does your app do?
This app combines **3 AI tools in one Streamlit app**:
- Summarize long text into short output
- Generate ideas from a topic
- Chatbot for Q&A with chat history  
Users can use the app by providing an API key in the sidebar or can use demo key then running any tool.

### 2) Which AI did you use and why?
I used **Groq** because it is fast, simple to integrate, and has strong free-tier friendly models for demos and testing.

### 3) How can someone run your app locally?
- Clone repo
- Install requirements
- Run the Streamlit app

Example:
```bash
git clone https://github.com/yskhub/multi-tool-ai-app.git
cd multi-tool-ai-app
pip install -r requirements.txt
streamlit run app.py

Also Users can directly use the app with below link

https://ysk-multi-tool-ai-app.streamlit.app/
