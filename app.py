import streamlit as st
import base64
import openai
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import speech_recognition as sr
import os
import hashlib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
AZURE_OPENAI_API_KEY1 = os.getenv("AZURE_OPENAI_API_KEY1")
AZURE_OPENAI_API_KEY2 = os.getenv("AZURE_OPENAI_API_KEY2")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")



# Streamlit UI setup
st.set_page_config(page_title="🎓CampusMate - KSSEM Assistant", layout="wide")


# Load FAISS index & text mappings
index = faiss.read_index("dataset/faiss_index_cleaned.bin")
with open("dataset/text_mappings_cleaned.json", "r", encoding="utf-8") as f:
    text_list = json.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to retrieve relevant chunks using FAISS
def retrieve_relevant_chunks(query, top_k=5):
    query_vector = np.array([model.encode(query)], dtype=np.float32)
    distances, indices = index.search(query_vector, top_k)
    return [text_list[i] for i in indices[0] if i < len(text_list)]

# Function to query Gemini API
def query_azure(user_query):
    context = "\n\n".join(retrieve_relevant_chunks(user_query))
    prompt = f"""You are an AI chatbot providing accurate information about an educational institution.
    Use the context below to answer the user's question:

    
    ### Context:
    {context}
    
    ### User Question:
    {user_query}
    
    ### Answer:"""
    client = openai.AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY2,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version="2024-02-01"
    )
    
    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT_NAME,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300

    )
    
    return response.choices[0].message.content if response.choices else "❌ I couldn't find an answer. Try rephrasing!"


# User Authentication
users_db = "users.json"

def load_users():
    if os.path.exists(users_db):
        with open(users_db, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(users_db, "w") as f:
        json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

users = load_users()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None

# Rotating KSSEM Logo
logo_path = "processed_logo.png"
st.sidebar.markdown(
    f"""
    <style>
        @keyframes rotate-earth {{
            0% {{ transform: rotateY(0deg); }}
            100% {{ transform: rotateY(360deg); }}
        }}
        .rotating-logo {{
            animation: rotate-earth 5s linear infinite;
            display: block;
            margin: auto;
            width: 130px;
        }}
    </style>
    <img src="data:image/jpeg;base64,{base64.b64encode(open(logo_path, 'rb').read()).decode()}" class="rotating-logo">
    """,
    unsafe_allow_html=True
)

# Login and Signup UI
# Login and Signup UI
if not st.session_state.authenticated:
    option = st.sidebar.radio("Login or Signup", ["Login", "Sign Up", "Continue as Guest"])
    
    if option == "Login":
        st.sidebar.subheader("Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            if username in users and users[username] == hash_password(password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.messages = users.get(username + "_history", [])
                st.sidebar.success("Logged in successfully!")
                st.rerun()
            else:
                st.sidebar.error("Invalid username or password")
    
    elif option == "Sign Up":
        st.sidebar.subheader("Create an Account")
        new_username = st.sidebar.text_input("New Username")
        new_password = st.sidebar.text_input("New Password", type="password")
        if st.sidebar.button("Sign Up"):
            if new_username in users:
                st.sidebar.error("Username already exists!")
            else:
                users[new_username] = hash_password(new_password)
                users[new_username + "_history"] = []
                save_users(users)
                st.sidebar.success("Account created successfully! Please login.")
    
    elif option == "Continue as Guest":
        st.session_state.authenticated = True
        st.session_state.username = "Guest"
        st.session_state.messages = []
        st.sidebar.info("You are using the chatbot as a guest. Chat history will not be saved.")
        st.rerun()

else:
    # Logout option
    if st.sidebar.button("Logout"):
        if st.session_state.username != "Guest":
            users[st.session_state.username + "_history"] = st.session_state.messages
            save_users(users)
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.messages = []
        st.rerun()

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
            audio = recognizer.listen(source, timeout=5)
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
user_query = None  # Ensure it is defined
st.markdown("""<div style='position: fixed; bottom: 10px; width: 100%;'>""", unsafe_allow_html=True)
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
    response = query_azure(user_query)
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
