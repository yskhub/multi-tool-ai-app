import os
import requests
import streamlit as st

APP_TITLE = "Multi-Tool AI App"
st.set_page_config(page_title=APP_TITLE, page_icon="🧰", layout="wide")

st.title("🧰 Multi-Tool AI App")
st.caption("Summarizer . Idea Generator . Simple Chatbot . User pastes API key manually")


# -----------------------------
# Provider Catalog (curated)
# -----------------------------
PROVIDERS = {
    "Groq": {
        "create_key_url": "https://console.groq.com/keys",
        "key_label": "Groq API key",
        "key_placeholder": "gsk_...",
        "models": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        "default_model": "llama-3.1-8b-instant",
        "steps": [
            "Open Groq keys page",
            "Log in / create account",
            "Click “Create API Key”",
            "Copy the key",
            "Paste it in the box below",
        ],
    },
    "Gemini (Google AI Studio)": {
        "create_key_url": "https://aistudio.google.com/app/apikey",
        "key_label": "Gemini API key",
        "key_placeholder": "AI Studio key",
        "models": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-2.0-flash"],
        "default_model": "gemini-2.5-flash",
        "steps": [
            "Open Google AI Studio API keys page",
            "Sign in with Google",
            "Create / view API key",
            "Copy the key",
            "Paste it in the box below",
        ],
    },
    "Mistral": {
        "create_key_url": "https://console.mistral.ai/api-keys/",
        "key_label": "Mistral API key",
        "key_placeholder": "mistral_...",
        "models": ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"],
        "default_model": "mistral-small-latest",
        "steps": [
            "Open Mistral Console",
            "Log in / create account",
            "Go to API Keys",
            "Create key and copy it",
            "Paste it in the box below",
        ],
    },
    "Hugging Face (HF Inference)": {
        "create_key_url": "https://huggingface.co/settings/tokens",
        "key_label": "Hugging Face token",
        "key_placeholder": "hf_...",
        "models": [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "google/gemma-2-9b-it",
        ],
        "default_model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "steps": [
            "Open Hugging Face tokens page",
            "Create a new token",
            "Copy the token",
            "Paste it in the box below",
        ],
    },
}

# Safe models ONLY for key testing (independent of Model Settings)
TEST_MODELS = {
    "Groq": "llama-3.1-8b-instant",
    "Gemini (Google AI Studio)": "gemini-2.5-flash",
    "Mistral": "mistral-small-latest",
    "Hugging Face (HF Inference)": "meta-llama/Meta-Llama-3-8B-Instruct",
}

# Admin key secret names (set in secrets.toml or Streamlit Cloud Secrets)
ADMIN_SECRET_NAMES = {
    "Groq": "GROQ_API_KEY",
    "Gemini (Google AI Studio)": "GEMINI_API_KEY",
    "Mistral": "MISTRAL_API_KEY",
    "Hugging Face (HF Inference)": "HF_API_KEY",
}


# -----------------------------
# Utilities
# -----------------------------
def system_prompt(tool: str) -> str:
    if tool == "summarizer":
        return "You summarize text accurately. Do not add facts."
    if tool == "ideas":
        return "You generate practical, creative ideas. Keep output structured."
    return "You are a helpful chatbot. Answer clearly and directly."


def call_groq(api_key: str, model: str, messages: list, temperature: float, max_tokens: int) -> str:
    from groq import Groq
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def call_gemini(api_key: str, model: str, prompt: str, temperature: float, max_tokens: int) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    return resp.text or ""


def call_mistral(api_key: str, model: str, messages: list, temperature: float, max_tokens: int) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Mistral error {r.status_code}: {r.text}")
    data = r.json()
    return data["choices"][0]["message"]["content"]


def call_hf(api_key: str, model: str, messages: list, temperature: float, max_tokens: int) -> str:
    from huggingface_hub import InferenceClient
    client = InferenceClient(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content


def run_llm(
    provider: str,
    api_key: str,
    model: str,
    tool: str,
    user_text: str,
    temperature: float,
    max_tokens: int,
    history=None,
) -> str:
    sys = system_prompt(tool)
    messages = [{"role": "system", "content": sys}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_text})

    if provider == "Groq":
        return call_groq(api_key, model, messages, temperature, max_tokens)

    if provider == "Mistral":
        return call_mistral(api_key, model, messages, temperature, max_tokens)

    if provider == "Hugging Face (HF Inference)":
        return call_hf(api_key, model, messages, temperature, max_tokens)

    if provider == "Gemini (Google AI Studio)":
        prompt = f"SYSTEM: {sys}\n"
        if history:
            for m in history:
                prompt += f"{m['role'].upper()}: {m['content']}\n"
        prompt += f"USER: {user_text}\nASSISTANT:"
        return call_gemini(api_key, model, prompt, temperature, max_tokens)

    raise ValueError("Unknown provider")


def safe_link_button(label: str, url: str) -> None:
    if hasattr(st, "link_button"):
        st.link_button(label, url, use_container_width=True)
    else:
        st.markdown(f"[{label}]({url})")


def get_saved_keys_for_provider(provider_name: str) -> dict:
    if "saved_api_keys" not in st.session_state:
        st.session_state.saved_api_keys = {}
    if provider_name not in st.session_state.saved_api_keys:
        st.session_state.saved_api_keys[provider_name] = {}
    return st.session_state.saved_api_keys[provider_name]


def get_admin_key(provider_name: str) -> str:
    secret_name = ADMIN_SECRET_NAMES.get(provider_name, "")
    if not secret_name:
        return ""

    # Try st.secrets first (Streamlit Cloud + local secrets.toml)
    try:
        if secret_name in st.secrets:
            return str(st.secrets[secret_name]).strip()
    except Exception:
        pass

    # Fallback: environment variable
    return os.getenv(secret_name, "").strip()


# -----------------------------
# Sidebar UI
# Order: provider -> key mode -> key box -> test -> create key+steps -> model settings
# -----------------------------
with st.sidebar:
    st.subheader("API Provider + Key")

    # session defaults
    if "provider_name" not in st.session_state:
        st.session_state.provider_name = "Groq"
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "model" not in st.session_state:
        st.session_state.model = PROVIDERS["Groq"]["default_model"]
    if "key_mode" not in st.session_state:
        st.session_state.key_mode = "Use admin key (demo)"

    # 1) Choose provider
    st.session_state.provider_name = st.selectbox(
        "Choose provider",
        list(PROVIDERS.keys()),
        index=list(PROVIDERS.keys()).index(st.session_state.provider_name),
    )
    p = PROVIDERS[st.session_state.provider_name]
    saved_keys = get_saved_keys_for_provider(st.session_state.provider_name)
    admin_key = get_admin_key(st.session_state.provider_name)

    st.divider()

    # 2) Key mode (admin demo / saved / paste)
    modes = ["Use admin key (demo)", "Use saved key", "Paste new key"]
    st.session_state.key_mode = st.radio("API Key option", modes, index=modes.index(st.session_state.key_mode))

    active_key = ""

    if st.session_state.key_mode == "Use admin key (demo)":
        if not admin_key:
            st.warning("Admin demo key is not configured for this provider. Choose 'Paste new key' instead.")
            active_key = ""
        else:
            st.success("Using admin demo key ✅ (users won’t see the key)")
            active_key = admin_key

    elif st.session_state.key_mode == "Use saved key":
        if len(saved_keys) == 0:
            st.info("No saved keys for this provider yet. Use 'Paste new key' and save one.")
            active_key = ""
        else:
            key_name = st.selectbox("Select saved key", list(saved_keys.keys()))
            active_key = saved_keys.get(key_name, "").strip()

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🗑️ Delete selected", use_container_width=True):
                    saved_keys.pop(key_name, None)
                    st.success("Deleted.")
                    st.rerun()
            with col_b:
                st.caption("Using saved key ✅")

    else:
        pasted = st.text_input(
            f"Paste your {p['key_label']}",
            value="",
            type="password",
            placeholder=p["key_placeholder"],
        )
        active_key = pasted.strip()

        with st.expander("Save this key (optional, session-only)"):
            nickname = st.text_input("Give it a name (example: Personal / Work)", value="")
            if st.button("💾 Save key", use_container_width=True, disabled=not (nickname.strip() and active_key)):
                saved_keys[nickname.strip()] = active_key
                st.success("Saved for this session.")
                st.rerun()

    # keep active key in session
    st.session_state.api_key = active_key

    # 3) Test API Key (safe provider test model always)
    test_clicked = st.button(
        "✅ Test API Key",
        use_container_width=True,
        disabled=not st.session_state.api_key.strip(),
    )
    if test_clicked:
        try:
            safe_test_model = TEST_MODELS.get(st.session_state.provider_name, p["default_model"])
            with st.spinner(f"Testing with {safe_test_model}…"):
                out = run_llm(
                    provider=st.session_state.provider_name,
                    api_key=st.session_state.api_key.strip(),
                    model=safe_test_model,
                    tool="chat",
                    user_text="Reply with exactly: OK",
                    temperature=0.0,
                    max_tokens=20,
                )
            st.success(f"API key works. Response: {out.strip()}")
        except Exception as e:
            st.error(f"Key test failed: {e}")

    # 4) Create API key + quick steps
    safe_link_button("⚡ Create API key", p["create_key_url"])
    with st.expander("How to create the key (quick steps)"):
        for i, step in enumerate(p["steps"], start=1):
            st.write(f"{i}. {step}")

    st.divider()

    # 5) Model Settings (for actual tools)
    st.subheader("Model Settings")

    if st.session_state.model not in p["models"]:
        st.session_state.model = p["default_model"]

    st.session_state.model = st.selectbox(
        "Choose model",
        p["models"],
        index=p["models"].index(st.session_state.model),
    )

    use_custom_model = st.toggle("Use custom model id (advanced)", value=False)
    if use_custom_model:
        st.session_state.model = st.text_input("Custom model id", value=st.session_state.model)

    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.1)
    max_tokens = st.slider("Max output tokens", 50, 2000, 600, 50)

    st.caption("Admin demo key uses your quota. Saved keys are session-only. Free tiers have limits.")


# -----------------------------
# Main App (3 tools)
# -----------------------------
provider = st.session_state.provider_name
api_key = st.session_state.api_key.strip()
model = st.session_state.model.strip()

if not api_key:
    st.info("Pick 'Use admin key (demo)' (if enabled) or paste your key in the sidebar.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📝 Text Summarizer", "💡 Idea Generator", "💬 Simple Chatbot"])

# Tool 1: Summarizer
with tab1:
    st.subheader("📝 Text Summarizer")
    text = st.text_area("Paste text → get short summary", height=220)

    summary_style = st.selectbox("Summary style", ["Bullet points", "Short paragraph"], index=0)
    length = st.slider("Length", 2, 10, 5)

    if st.button("Summarize", use_container_width=True, disabled=not text.strip()):
        instruction = (
            f"Summarize into exactly {length} bullet points."
            if summary_style == "Bullet points"
            else f"Summarize in about {length} sentences."
        )
        prompt = f"{instruction}\n\nTEXT:\n{text.strip()}"

        try:
            with st.spinner("Summarizing…"):
                summary = run_llm(provider, api_key, model, "summarizer", prompt, temperature, max_tokens)
            st.markdown("### ✅ Summary")
            st.write(summary)
        except Exception as e:
            st.error(str(e))

# Tool 2: Idea Generator
with tab2:
    st.subheader("💡 Idea Generator")
    topic = st.text_input("Enter topic → get ideas", placeholder="Example: Instagram reels for a coffee shop")
    idea_count = st.slider("How many ideas?", 3, 20, 10, 1)

    if st.button("Generate ideas", use_container_width=True, disabled=not topic.strip()):
        prompt = f"Give me {idea_count} strong ideas about: {topic.strip()}. Number them and add 1-line detail each."
        try:
            with st.spinner("Generating…"):
                ideas = run_llm(provider, api_key, model, "ideas", prompt, temperature, max_tokens)
            st.markdown("### ✅ Ideas")
            st.write(ideas)
        except Exception as e:
            st.error(str(e))

# Tool 3: Chatbot
with tab3:
    st.subheader("💬 Simple Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    col1, _ = st.columns([1, 1])
    with col1:
        if st.button("🧹 Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    user_msg = st.chat_input("Ask questions → get AI answers")
    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})

        recent = [m for m in st.session_state.chat_history if m["role"] in ("user", "assistant")][-12:]

        try:
            with st.spinner("Thinking…"):
                reply = run_llm(
                    provider=provider,
                    api_key=api_key,
                    model=model,
                    tool="chat",
                    user_text=user_msg,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    history=recent[:-1],
                )
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()
        except Exception as e:
            st.error(str(e))
