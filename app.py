import streamlit as st
from groq import Groq

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Multi-Tool AI App", page_icon="🧰", layout="centered")
st.title("🧰 Multi-Tool AI App")
st.caption("Text Summarizer . Idea Generator . Simple Chatbot (User enters Groq API key manually)")

# ----------------------------
# Sidebar: API key + settings
# ----------------------------
with st.sidebar:
    st.header("🔑 API Key")

    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = ""

    st.session_state.groq_api_key = st.text_input(
        "Paste your Groq API key",
        value=st.session_state.groq_api_key,
        type="password",
        placeholder="gsk_...",
        help="Paste your Groq API key here. It stays only in your current session.",
    )

    # ✅ Test API key button
    if st.button("✅ Test API Key", use_container_width=True):
        key = st.session_state.groq_api_key.strip()
        if not key:
            st.warning("Paste an API key first.")
        else:
            try:
                test_client = Groq(api_key=key)
                resp = test_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                    max_tokens=10,
                    temperature=0.0,
                )
                out = resp.choices[0].message.content.strip()
                if "OK" in out:
                    st.success("API key works. OK")
                else:
                    st.success(f"API key works. Response: {out}")
            except Exception as e:
                st.error(f"Key test failed: {e}")

    st.divider()
    st.header("⚙️ Settings")
    model = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        index=0,
    )
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.1)

api_key = st.session_state.groq_api_key.strip()
if not api_key:
    st.warning("Enter your Groq API key in the sidebar to start using the tools.")
    st.stop()

# Create client per user session (do NOT cache globally)
client = Groq(api_key=api_key)

def groq_chat(messages, max_tokens=700):
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


# ----------------------------
# Tabs: 3 tools
# ----------------------------
tab1, tab2, tab3 = st.tabs(["📝 Summarizer", "💡 Idea Generator", "🤖 Chatbot"])

# Tool 1: Summarizer
with tab1:
    st.subheader("📝 Text Summarizer")
    text = st.text_area("Paste text", height=220)

    style = st.selectbox("Summary style", ["Bullet points", "Short paragraph"], index=0)
    length = st.slider("Length", 2, 10, 5)

    if st.button("Summarize", use_container_width=True):
        if not text.strip():
            st.warning("Paste some text first.")
        else:
            instruction = (
                f"Summarize into exactly {length} bullet points."
                if style == "Bullet points"
                else f"Summarize in about {length} sentences."
            )

            with st.spinner("Summarizing..."):
                summary = groq_chat(
                    [
                        {"role": "system", "content": "Summarize accurately without adding new facts."},
                        {"role": "user", "content": f"{instruction}\n\nTEXT:\n{text}"},
                    ],
                    max_tokens=500,
                )

            st.markdown("### ✅ Summary")
            st.write(summary)

# Tool 2: Idea Generator
with tab2:
    st.subheader("💡 Idea Generator")
    topic = st.text_input("Enter topic", placeholder="Example: content ideas for a coffee shop")
    idea_count = st.slider("Ideas", 3, 20, 10)

    if st.button("Generate ideas", use_container_width=True):
        if not topic.strip():
            st.warning("Enter a topic first.")
        else:
            with st.spinner("Generating ideas..."):
                ideas = groq_chat(
                    [
                        {"role": "system", "content": "Generate practical, creative ideas. Keep them structured."},
                        {"role": "user", "content": f"Give me {idea_count} ideas about: {topic}. Number them and add 1-line detail each."},
                    ],
                    max_tokens=700,
                )

            st.markdown("### ✅ Ideas")
            st.write(ideas)

# Tool 3: Chatbot
with tab3:
    st.subheader("🤖 Simple Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "system", "content": "You are a helpful assistant. Answer clearly and directly."}
        ]

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.chat_history = [
            {"role": "system", "content": "You are a helpful assistant. Answer clearly and directly."}
        ]
        st.rerun()

    # display chat (skip system)
    for msg in st.session_state.chat_history:
        if msg["role"] in ("user", "assistant"):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    user_input = st.chat_input("Ask me anything...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # keep history short to reduce token usage
        system_msg = st.session_state.chat_history[0:1]
        recent_msgs = [m for m in st.session_state.chat_history[1:] if m["role"] in ("user", "assistant")][-12:]
        messages = system_msg + recent_msgs

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = groq_chat(messages, max_tokens=700)
                st.write(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
