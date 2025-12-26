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

## 📂 Project Structure

