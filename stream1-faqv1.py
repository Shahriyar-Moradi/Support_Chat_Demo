import time
import streamlit as st
# import PyPDF2
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import os
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
import os
from openai import OpenAI
import base64
# Load environment variables
load_dotenv() 

# INDEX_NAME = "abanprime-chat" 

INDEX_NAME = "aban-rag-ft2-faq-v1-1" 

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# (Assumes the index already exists.)
while not pc.describe_index(INDEX_NAME).status.get('ready', False):
    time.sleep(1)
index = pc.Index(INDEX_NAME)

ft_model = SentenceTransformer("/home/shahriar/Work/AbanTether/raggpt/pincone_RAG/embedded_fintune/distilroberta-ai-faq1-v1-embeddingfinetuned-model-1")
print("Fine-tuned model dimension:", ft_model.get_sentence_embedding_dimension())

# def get_embedding(text: str, model: str = "text-embedding-3-large") -> list:
#     """Generates an embedding for the given text using OpenAI."""
#     response = client.embeddings.create(input=[text], model=model)
#     return response.data[0].embedding

def get_embedding(text: str) -> list:
    emb = ft_model.encode(text)
    return emb.tolist()

def query_index(query: str) -> str:
    """
    Queries the Pinecone index with the query's embedding and 
    returns concatenated context from the top matching chunks.
    """

    query_emb = get_embedding(query, 
                            # model=ft_model,
                            # model="text-embedding-3-large",
                            )
    results = index.query(
        # namespace="ns1",
        namespace="faq-namespace",
        vector=query_emb,
        top_k=3,
        include_values=False,
        include_metadata=True
    )
    print("Results:", results)
    retrieved_texts = [match["metadata"]["text"] for match in results.get("matches", [])]
    # retrieved_texts = [match["metadata"]["answer"] for match in results.get("matches", [])]
    
    # retrieved_texts = []
    # for match in results.get("matches", []):
    #     metadata = match["metadata"]
    
    # # If it's a FAQ entry, use question + answer
    #     if "question" in metadata and "answer" in metadata:
    #         retrieved_texts.append(f"Q: {metadata['question']}\nA: {metadata['answer']}")
        
    #     # If it's a PDF entry, use 'text'
    #     elif "text" in metadata:
    #         retrieved_texts.append(metadata["text"])
        
    #     # If no known metadata keys, add a placeholder
    #     else:
    #         retrieved_texts.append("[No text available]")
    
    return "\n\n".join(retrieved_texts)

def generate_response(query: str, context: str,conversation_history: str) -> str:
    """
    Builds a prompt with the retrieved context and generates a conversational
    answer using GPT-4 (or GPT-4o).
    """
    prompt = (
        "You are a friendly, helpful, and natural conversational AI assistant.work in exchange company named Abanprime and in persian is آبان پرایم you should answer question in supportive way Based on the following context extracted from your indexed documents, "
         "Engage in a natural conversation,answer each question and procedure of any buying or selling step by step by creating a diagram of procedure"
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
        # model="o1",
        # model="gpt-4o",
        model="gpt-4.5-preview",
        # Change to "gpt-4" if preferred
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return chat_response.choices[0].message.content

# ----------------------------------
# Streamlit Chat Interface
# ----------------------------------
st.title("AbanPrime Chat Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Text input for the user's query.
user_query = st.text_input("Ask a question:")

if st.button("Send") and user_query:
    with st.spinner("Generating response..."):
        # Query the index to get relevant context.
        context = query_index(user_query)
        # Build conversation history as a string.
        conversation_history = "\n".join(
            [f"{speaker}: {message}" for speaker, message in st.session_state.chat_history]
        )
        # Generate the assistant's response.
        answer = generate_response(user_query, context, conversation_history)
        # Append the new exchange to the chat history.
        st.session_state.chat_history.append(("User", user_query))
        st.session_state.chat_history.append(("Assistant", answer))

# Display the conversation history.
for speaker, message in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {message}")