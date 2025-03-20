import streamlit as st
import base64
import google.generativeai as genai
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Streamlit UI setup
st.set_page_config(page_title="🎓𝙲𝚊𝚖𝚙𝚞𝚜𝙼𝚊𝚝𝚎", layout="wide")

# Configure Gemini API
genai.configure(api_key="AIzaSyChUzmOrRlZRCtmY7nv90suM86bcUj1z58")  # Replace with your actual key

# Load FAISS index & text mappings
index = faiss.read_index("faiss_index_cleaned.bin")
with open("text_mappings_cleaned.json", "r", encoding="utf-8") as f:
    text_list = json.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to retrieve relevant chunks using FAISS
def retrieve_relevant_chunks(query, top_k=3):
    query_vector = np.array([model.encode(query)], dtype=np.float32)
    distances, indices = index.search(query_vector, top_k)
    return [text_list[i] for i in indices[0]]

# Function to query Gemini API
def query_gemini(user_query):
    context = "\n\n".join(retrieve_relevant_chunks(user_query))
    prompt = f"""You are an AI chatbot providing accurate information about an educational institution.
    Use the context below to answer the user's question:

    ### Context:
    {context}

    ### User Question:
    {user_query}

    ### Answer:"""

    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)

    return response.text if response else "❌ I couldn't find an answer. Try rephrasing!"

# Set KSSEM logo as background
def set_background(image_path):
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background: url("data:image/jpg;base64,{base64_str}") no-repeat center center fixed;
        background-size: 60%;
    }}
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255, 255, 255, 0.95);
        z-index: -1;
    }}
    
    /* Chat container */
    .chat-container {{
        max-width: 800px;
        margin: auto;
        display: flex;
        flex-direction: column;
    }}

    /* Message Wrapper */
    .message-wrapper {{
        display: flex;
        width: 100%;
        margin: 5px 0;
    }}

    /* User messages (Right Side) */
    .user-message {{
        justify-content: flex-end;
    }}

    .user-msg {{
        background-color: #D0EFFF;  /* Light blue */
        color: black;
        padding: 12px 16px;
        border-radius: 18px;
        max-width: 70%;
        word-wrap: break-word;
        display: inline-block;
    }}

    /* Bot messages (Left Side) */
    .bot-message {{
        justify-content: flex-start;
    }}

    .bot-msg {{
        background-color: #E0E0E0;  /* Light grey */
        color: black;
        padding: 12px 16px;
        border-radius: 18px;
        max-width: 70%;
        word-wrap: break-word;
        display: inline-block;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("kssem_logo.jpg")

# Move title slightly upward
st.markdown(
    '<h1 style="text-align: center; margin-top: -80px;">🎓CampusMate- Kssem Assistant</h1>',
    unsafe_allow_html=True
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat display container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f'''
            <div class="message-wrapper user-message">
                <div class="chat-bubble user-msg">{message["content"]}</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'''
            <div class="message-wrapper bot-message">
                <div class="chat-bubble bot-msg">{message["content"]}</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

st.markdown('</div>', unsafe_allow_html=True)

# User input field
if user_query := st.chat_input("Ask me anything about your institution..."):
    # Save user question
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Display user message immediately on the right
    st.markdown(
        f'''
        <div class="message-wrapper user-message">
            <div class="chat-bubble user-msg">{user_query}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    # Generate bot response
    response = query_gemini(user_query)

    # Save bot response
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display chatbot response on the left
    st.markdown(
        f'''
        <div class="message-wrapper bot-message">
            <div class="chat-bubble bot-msg">{response}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

# Sidebar for additional settings and chat history
with st.sidebar:
    st.header("⚙ Settings")
    
    # Button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # Display chat history under settings
    st.header("📜 Chat History")
    for message in st.session_state.messages:
        role = "🧑‍💻 You: " if message["role"] == "user" else "🤖 Bot: "
        st.markdown(f"{role} {message['content']}", unsafe_allow_html=True)
