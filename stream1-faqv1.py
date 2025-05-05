import time
import uuid
from datetime import datetime
import streamlit as st
import os

from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# ----------------------------------
# Application Setup & Config
# ----------------------------------
st.set_page_config(
    page_title="AbanPrime AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# ----------------------------------
# Database Setup using SQLAlchemy
# ----------------------------------
DATABASE_URL = "sqlite:///./chat_history.db"  # SQLite file-based database

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(String, index=True, nullable=False)
    speaker = Column(String, nullable=False)  # "User" or "Assistant"
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ----------------------------------
# API Client Initialization
# ----------------------------------
load_dotenv()

# Create OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Set index name
INDEX_NAME = "aban-rag-persian-faq-v1-3-21thapril"

# Initialize the index
# @st.cache_resource
while not pc.describe_index(INDEX_NAME).status.get('ready', False):
    time.sleep(1)
index = pc.Index(INDEX_NAME)

# Initialize the SentenceTransformer with the fine-tuned model
@st.cache_resource
def load_sentence_transformer():
    model = SentenceTransformer("Shahriardev/distobert-finetuned-embedding-faq1-v1-1")
    return model

ft_model = load_sentence_transformer()

# ----------------------------------
# Helper Functions
# ----------------------------------
def get_embedding(text: str) -> list:
    """Generates an embedding for the given text using the fine-tuned model."""
    emb = ft_model.encode(text)
    return emb.tolist()

def query_index(query: str) -> str:
    """
    Queries the Pinecone index with the query's embedding and returns
    concatenated context from the top matching chunks.
    """
    query_emb = get_embedding(query)
    results = index.query(
        namespace="faq-namespace",
        vector=query_emb,
        top_k=3,
        include_values=False,
        include_metadata=True
    )
    st.session_state.debug_info = f"Query Results: {results}"
    retrieved_texts = [match["metadata"]["text"] for match in results.get("matches", [])]
    return "\n\n".join(retrieved_texts)

def generate_response(query: str, context: str, conversation_history: str) -> str:
    """
    Builds a prompt with the retrieved context and generates a conversational answer
    using GPT-4o.
    """
    prompt = (
        "You are a friendly, helpful, and natural conversational AI assistant. Work in exchange company named Abanprime and in persian is آبان پرایم. "
        "You should answer questions in a supportive way based on the following context extracted from your indexed documents and don't answer not relevant question. "
        "Engage in a natural conversation, and if the user mentions topics such as buying Tether (USDT), "
        "ask follow-up clarifying questions like 'How much USDT would you like to buy?' and "
        "'How will you pay? In which currency?'.\n\n"
        "Context from the vector database:\n"
        f"{context}\n\n"
        "Conversation so far:\n"
        f"{conversation_history}\n"
        f"User: {query}\n"
        "Assistant:"
    )

    chat_response = client.chat.completions.create(
        model="chatgpt-4o-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000
    )
    return chat_response.choices[0].message.content

def get_db():
    """Gets a database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

def load_conversation_history(chat_id: str, db: Session):
    """Retrieve conversation history for a given chat_id from the database."""
    messages = db.query(ChatMessage).filter(ChatMessage.chat_id == chat_id).order_by(ChatMessage.created_at).all()
    return [(msg.speaker, msg.message) for msg in messages]

def save_message(chat_id: str, speaker: str, message: str, db: Session) -> None:
    """Save a chat message to the database."""
    new_msg = ChatMessage(chat_id=chat_id, speaker=speaker, message=message)
    db.add(new_msg)
    db.commit()

# ----------------------------------
# Streamlit UI Functions
# ----------------------------------
def display_message(role, content):
    """Display a message with appropriate styling."""
    if role == "User":
        st.markdown(f"<div style='background-color: #e6f7ff; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>You:</strong> {content}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>Assistant:</strong> {content}</div>", unsafe_allow_html=True)

def get_chat_id():
    """Get or create a chat ID for the current session."""
    if "chat_id" not in st.session_state:
        st.session_state["chat_id"] = str(uuid.uuid4())
    return st.session_state["chat_id"]

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "debug_info" not in st.session_state:
        st.session_state.debug_info = ""
    
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False

# ----------------------------------
# Main Application
# ----------------------------------
def main():
    st.title("AbanPrime AI Assistant")
    
    # Initialize session state
    initialize_session_state()
    
    # Get chat ID
    chat_id = get_chat_id()
    
    # Debug toggle
    st.sidebar.title("Debug Options")
    st.session_state.debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
    
    # Sidebar with information
    st.sidebar.title("About")
    st.sidebar.info(
        "This is an AI assistant for AbanPrime. "
        "Ask questions about exchange services, buying/selling USDT, "
        "or any other services offered by AbanPrime."
    )
    
    # Clear chat button
    if st.sidebar.button("Start New Chat"):
        st.session_state.chat_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.experimental_rerun()
    
    # Display debug info
    if st.session_state.debug_mode:
        st.sidebar.subheader("Debug Information")
        st.sidebar.text(f"Chat ID: {chat_id}")
        st.sidebar.text_area("Query Results", st.session_state.debug_info, height=200)
    
    # Load chat history from database on first load
    if not st.session_state.messages:
        with st.spinner("Loading conversation history..."):
            db = get_db()
            st.session_state.messages = load_conversation_history(chat_id, db)
    
    # Display existing messages
    for speaker, message in st.session_state.messages:
        display_message(speaker, message)
    
    # Input for new message
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Show user message
        display_message("User", user_input)
        
        # Add to session state
        st.session_state.messages.append(("User", user_input))
        
        # Save user message to database
        db = get_db()
        save_message(chat_id, "User", user_input, db)
        
        # Process the response
        with st.spinner("Thinking..."):
            # Format conversation history
            conversation_history_str = "\n".join(
                [f"{speaker}: {msg}" for speaker, msg in st.session_state.messages]
            )
            
            # Query vector store for context
            context = query_index(user_input)
            
            # Generate response
            response = generate_response(user_input, context, conversation_history_str)
            
            # Save assistant message to database
            save_message(chat_id, "Assistant", response, db)
            
            # Add to session state
            st.session_state.messages.append(("Assistant", response))
            
            # Display assistant message
            display_message("Assistant", response)

if __name__ == "__main__":
    main()
