import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Mini LLM")
st.title("Mini LLM Chat")

st.sidebar.header("Settings")
steps       = st.sidebar.slider("Steps", 10, 300, 100)
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7)
mode        = st.sidebar.radio("Mode", ["RAG Generate", "Plain Generate"])

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Type your message...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Generating..."):
        try:
            if mode == "RAG Generate":
                res = requests.post(f"{API_URL}/rag-generate", json={
                    "query": prompt,
                    "steps": steps
                })
            else:
                res = requests.post(f"{API_URL}/generate", json={
                    "prompt":      prompt,
                    "steps":       steps,
                    "temperature": temperature
                })

            if res.status_code == 200:
                reply = res.json().get("generated", "")
            else:
                reply = f"API error {res.status_code}: {res.text}"

        except requests.exceptions.ConnectionError:
            reply = "Cannot connect to API. Is FastAPI running?"

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)