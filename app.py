import streamlit as st
import base64
import google.generativeai as genai
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import speech_recognition as sr

# Streamlit UI setup
st.set_page_config(page_title="🎓CampusMate - KSSEM Assistant", layout="wide")

# Configure Gemini API (Use environment variables instead of hardcoding API keys)
genai.configure(api_key="AIzaSyChUzmOrRlZRCtmY7nv90suM86bcUj1z58")  # Replace with a secure key method

# Load FAISS index & text mappings
index = faiss.read_index("dataset/faiss_index_cleaned.bin")
with open("dataset/text_mappings_cleaned.json", "r", encoding="utf-8") as f:
    text_list = json.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to retrieve relevant chunks using FAISS
def retrieve_relevant_chunks(query, top_k=10):
    query_vector = np.array([model.encode(query)], dtype=np.float32)
    distances, indices = index.search(query_vector, top_k)
    return [text_list[i] for i in indices[0] if i < len(text_list)]

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
    
    model = genai.GenerativeModel("gemini-1.5-pro",generation_config={"temperature": 0.3})

    response = model.generate_content(prompt)
    return response.text if response and response.text else "❌ I couldn't find an answer. Try rephrasing!"

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
    
    .chat-container {{
        max-width: 800px;
        margin: auto;
        display: flex;
        flex-direction: column;
    }}

    .message-wrapper {{
        display: flex;
        width: 100%;
        margin: 5px 0;
    }}

    .user-message {{
        justify-content: flex-end;
    }}

    .user-msg {{
        background: linear-gradient(135deg, #4A90E2, #007BFF);
        color: white;
        padding: 12px 16px;
        border-radius: 20px;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        display: inline-block;
    }}

    .bot-message {{
        justify-content: flex-start;
    }}

    .bot-msg {{
        background: linear-gradient(135deg, #E0E0E0, #F5F5F5);
        color: black;
        padding: 12px 16px;
        border-radius: 20px;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        display: inline-block;
    }}
    .stButton > button {{
        background: #007BFF;
        color: white;
        border-radius: 5px;
    }}
    .stButton > button:hover {{
        background: #0056b3;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("kssem_logo.jpg")

# Move title slightly upward
st.markdown(
    '<h1 style="text-align: center; margin-top: -80px;">🎓CampusMate - KSSEM Assistant</h1>',
    unsafe_allow_html=True
)

# Function for voice search
def voice_search():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=10)
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Sorry, could not understand the audio.")
        except sr.RequestError:
            st.error("Could not request results. Check your internet connection.")
        except sr.WaitTimeoutError:
            st.error("Listening timed out. Please try again.")
    return ""

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat display container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="message-wrapper user-message"><div class="chat-bubble user-msg">{message["content"]}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="message-wrapper bot-message"><div class="chat-bubble bot-msg">{message["content"]}</div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Move text input and voice search to the bottom
col1, col2 = st.columns([10, 1])

with col1:
    user_query = st.chat_input("Ask me anything about your institution...")

with col2:
    if st.button("🎙️"):
        voice_input = voice_search()
        if voice_input:
            user_query = voice_input  # Assign voice result to user query
st.markdown("""</div>""", unsafe_allow_html=True)

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    response = query_gemini(user_query)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Sidebar for settings and chat history
with st.sidebar:
    st.header("⚙ Settings")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.header("📜 Chat History")
    for message in st.session_state.messages:
        role = "🧑‍💻 You: " if message["role"] == "user" else "🤖 Bot: "
        st.markdown(f"{role} {message['content']}", unsafe_allow_html=True)