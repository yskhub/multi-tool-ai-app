import streamlit as st
from groq import Groq

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Multi-Tool AI App", page_icon="🧰", layout="centered")
st.title("🧰 Multi-Tool AI App")
st.caption("Summarizer . Idea Generator . Simple Chatbot")

# ----------------------------
# Helpers
# ----------------------------
def ui_link_button(label: str, url: str) -> None:
    """Streamlit link button with a safe fallback for older Streamlit versions."""
    if hasattr(st, "link_button"):
        st.link_button(label, url, use_container_width=True)
    else:
        st.markdown(f"[{label}]({url})")

PROVIDERS = {
    "Groq": {
        "create_key_url": "https://console.groq.com/keys",
        "key_label": "Groq API key",
        "key_hint": "gsk_...",
        "steps": [
            "Open Groq API Keys page",
            "Log in / create account",
            "Click “Create API Key”",
            "Copy the key",
            "Paste it in the box above",
        ],
    },
    "Google AI Studio (Gemini)": {
        "create_key_url": "https://aistudio.google.com/app/apikey",
        "key_label": "Gemini API key",
        "key_hint": "AI Studio key",
        "steps": [
            "Open Google AI Studio API key page",
            "Sign in with Google",
            "Create / view API key",
            "Copy the key",
            "Paste it in the box above",
        ],
    },
    "OpenAI": {
        "create_key_url": "https://platform.openai.com/api-keys",
        "key_label": "OpenAI API key",
        "key_hint": "sk-...",
        "steps": [
            "Open OpenAI API keys page",
            "Log in",
            "Create a new API key",
            "Copy it (you may not see it again)",
            "Paste it in the box above",
        ],
    },
    "Anthropic (Claude)": {
        "create_key_url": "https://console.anthropic.com/",
        "key_label": "Anthropic API key",
        "key_hint": "sk-ant-...",
        "steps": [
            "Open Anthropic Console",
            "Log in / create account",
            "Go to “API Keys” in the console",
            "Create a key and copy it",
            "Paste it in the box above",
        ],
    },
    "Mistral": {
        "create_key_url": "https://console.mistral.ai/",
        "key_label": "Mistral API key",
        "key_hint": "mistral_...",
        "steps": [
            "Open Mistral Console",
            "Log in / create account",
            "Go to Workspace → API keys",
            "Create a new key and copy it",
            "Paste it in the box above",
        ],
    },
    "Cohere": {
        "create_key_url": "https://dashboard.cohere.com/api-keys",
        "key_label": "Cohere API key",
        "key_hint": "co_...",
        "steps": [
            "Open Cohere API Keys page",
            "Log in / create account",
            "Create a Trial/Production key",
            "Copy the key",
            "Paste it in the box above",
        ],
    },
    "Together.ai": {
        "create_key_url": "https://api.together.xyz/settings/api-keys",
        "key_label": "Together API key",
        "key_hint": "together_...",
        "steps": [
            "Open Together settings → API Keys",
            "Log in / create account",
            "Create a new key",
            "Copy the key",
            "Paste it in the box above",
        ],
    },
    "OpenRouter": {
        "create_key_url": "https://openrouter.ai/keys",
        "key_label": "OpenRouter API key",
        "key_hint": "sk-or-...",
        "steps": [
            "Open OpenRouter Keys page",
            "Log in / create account",
            "Create a key (optional credit limit)",
            "Copy the key",
            "Paste it in the box above",
        ],
    },
}

# ----------------------------
# Sidebar UI
# ----------------------------
with st.sidebar:
    st.header("🔐 API Provider + Key")

    if "provider_name" not in st.session_state:
        st.session_state.provider_name = "Groq"
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    st.session_state.provider_name = st.selectbox(
        "Choose provider",
        list(PROVIDERS.keys()),
        index=list(PROVIDERS.keys()).index(st.session_state.provider_name),
    )
    provider = PROVIDERS[st.session_state.provider_name]

    ui_link_button(f"⚡ Create {st.session_state.provider_name} API Key", provider["create_key_url"])

    with st.expander("How to create the key (quick steps)"):
        for i, step in enumerate(provider["steps"], start=1):
            st.write(f"{i}. {step}")
        st.caption("Heads up: “Free key” usually means free-tier/limited usage, not unlimited forever.")

    st.divider()

    st.session_state.api_key = st.text_input(
        f"Paste your {provider['key_label']}",
        value=st.session_state.api_key,
        type="password",
        placeholder=provider["key_hint"],
    )

    # ✅ Test API key button
    if st.button("✅ Test API Key", use_container_width=True):
        key = st.session_state.api_key.strip()

        if not key:
            st.warning("Paste an API key first.")
        elif st.session_state.provider_name != "Groq":
            st.info("This demo app runs the tools using Groq only. Switch provider to Groq to test inside the app.")
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
                if out == "OK":
                    st.success("API key works. OK")
                else:
                    st.success(f"API key works. Response: {out}")
            except Exception as e:
                st.error(f"Key test failed: {e}")

    st.divider()
    st.header("⚙️ Model Settings (Groq)")
    model = st.selectbox("Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"], index=0)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.1)

# ----------------------------
# Enforce provider for this demo
# ----------------------------
if st.session_state.provider_name != "Groq":
    st.info(
        "You can use the sidebar to create API keys from many providers.\n\n"
        "But this demo app currently runs the AI tools using **Groq** only.\n"
        "Switch provider to **Groq** to use Summarizer / Idea Generator / Chatbot."
    )
    st.stop()

api_key = st.session_state.api_key.strip()
if not api_key:
    st.warning("Enter your Groq API key in the sidebar to start using the tools.")
    st.stop()

# Create Groq client for this user session (no global cache)
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
    topic = st.text_input("Enter topic", placeholder="Example: Instagram reels for a coffee shop")
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

    for msg in st.session_state.chat_history:
        if msg["role"] in ("user", "assistant"):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    user_input = st.chat_input("Ask me anything...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Keep history short to reduce usage
        system_msg = st.session_state.chat_history[0:1]
        recent_msgs = [m for m in st.session_state.chat_history[1:] if m["role"] in ("user", "assistant")][-12:]
        messages = system_msg + recent_msgs

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = groq_chat(messages, max_tokens=700)
                st.write(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
