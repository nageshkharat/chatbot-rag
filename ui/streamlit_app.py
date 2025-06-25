import streamlit as st
import requests

st.title("📚 Document Chatbot")
st.markdown("---")
st.header("Upload a Document")

# Upload file section
uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
if uploaded_file is not None:
    with st.spinner("Uploading..."):
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        try:
            response = requests.post("http://localhost:8001/upload", files=files)
            response.raise_for_status()
            st.success("File uploaded! It will be processed and indexed.")
        except Exception as e:
            st.error(f"Upload failed: {e}")

st.markdown("---")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for q, a in st.session_state.history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")

# Handle chat input
def handle_query():
    query = st.session_state.query_input.strip()
    if not query:
        return

    try:
        response = requests.post("http://localhost:8001/chat", json={"query": query})
        response.raise_for_status()
        answer = response.json()["answer"]

        st.session_state.history.append((query, answer))
        st.session_state.query_input = ""  # clear input after submitting

    except Exception as e:
        st.error(f"Error: {e}")

# Input box with callback
st.text_input("Enter your question:", key="query_input", on_change=handle_query)
