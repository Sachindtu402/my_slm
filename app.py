import streamlit as st
import torch
import json
import tiktoken
import time
from huggingface_hub import hf_hub_download
from model import GPT, GPTConfig

# ---------- Page Config (MUST BE FIRST) ----------
st.set_page_config(page_title="MySLM Chat", layout="centered")

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    hf_token = st.secrets["HF_TOKEN"]

    model_path = hf_hub_download(
        repo_id="Sachin0803/myslm",
        filename="model.pt",
        token=hf_token
    )

    config_path = hf_hub_download(
        repo_id="Sachin0803/myslm",
        filename="config.json",
        token=hf_token
    )

    with open(config_path) as f:
        cfg = json.load(f)

    model = GPT(GPTConfig(**cfg))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    return model, enc


model, enc = load_model()

# ---------- Generate ----------
def generate(prompt, max_new_tokens=80, temperature=0.7):
    tokens = enc.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

    return enc.decode(out[0].tolist())


# ---------- UI ----------
st.title("💬 Chat with MySLM")

st.markdown("""
### 🧸 Kids Story Chatbot  

This chatbot is specially designed for **children**.

It can:
- 📖 Tell simple and fun stories  
- 🐱 Create stories about animals, friends, and adventures  
- 😊 Use easy and friendly language  

✨ Try asking:  
*"Tell me a story about a cat"*  
""")

st.info("✨ Kid-friendly AI that creates fun and simple stories!")

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Settings")

temperature = st.sidebar.slider(
    "Temperature", 0.1, 1.5, 0.7, key="temp"
)

max_tokens = st.sidebar.slider(
    "Max Tokens", 20, 200, 80, key="tokens"
)

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.messages = []

# ---------- Session ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Chat Display ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Input ----------
user_input = st.chat_input("Ask for a story...", key="chat_input_main")

if user_input:
    # show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    prompt = f"Write a simple story:\n{user_input}\nStory:"

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = generate(prompt, max_tokens, temperature)
            except Exception as e:
                reply = f"Error: {str(e)}"

        # clean output
        reply = reply.replace("Assistant:", "").replace("User:", "").strip()

        # typing effect
        placeholder = st.empty()
        typed = ""

        for ch in reply:
            typed += ch
            placeholder.markdown(typed)
            time.sleep(0.01)

    st.session_state.messages.append({"role": "assistant", "content": reply})
