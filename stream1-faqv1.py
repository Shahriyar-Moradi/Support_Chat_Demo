import time
import streamlit as st
import requests
import json
from pinecone import Pinecone
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from typing import Literal, List, Dict, Union, Optional, Any
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import re
import uuid
from datetime import datetime

# Import the memory pipeline functions
from memory_pipeline import (
    initialize_memory,
    process_conversation_with_memory,
    ConversationMemory,
    add_message_to_memory,
    extract_entities,
    generate_conversation_summary,
    prepare_transaction_confirmation,
    create_transaction_summary
)

# Load environment variables
load_dotenv()

# Configuration
INDEX_NAME = "aban-rag-ft2-faq-v1-1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
# Wait for index to be ready
while not pc.describe_index(INDEX_NAME).status.get('ready', False):
    time.sleep(1)
index = pc.Index(INDEX_NAME)

# Load fine-tuned embedding model
# ft_model = SentenceTransformer("/Users/shahriar/Desktop/Work/AbanTether/AbanTether/raggpt/pincone_RAG/embedded_fintune/distobert-finetuned-embedding-faq1-v1-1")
# ft_model = SentenceTransformer("Shahriardev/distobert-finetuned-embedding-faq1-v1-1",device='meta')
# print("Fine-tuned model dimension:", ft_model.get_sentence_embedding_dimension())


import torch

# low‑memory, meta‑safe instantiation:
ft_model = SentenceTransformer(
    "Shahriardev/distobert-finetuned-embedding-faq1-v1-1",
    # device_map={"": "cpu"},       # pin all sub‑modules to CPU
    low_cpu_mem_usage=True,       # do init_empty_weights + to_empty
    torch_dtype=torch.float32,    # or float16 if you prefer
    trust_remote_code=True        # if the repo defines custom layers
)

print("Fine‑tuned model dimension:", ft_model.get_sentence_embedding_dimension())


# Currency mappings and keywords
CURRENCY_KEYWORDS = {
    "IRR": ["تومان", "تومن", "تومنی", "ریال", "ریالی", "irt", "irr"],
    "AED": ["درهم", "درهم امارات", "درهم اماراتی", "aed", "امارات"],
    "USDT": ["تتر", "یو‌اس‌دی‌تی", "usdt", "تتری", "دلار"]
}

TRANSACTION_KEYWORDS = {
    "buy": ["خرید", "بخرم", "میخرم", "می‌خرم", "خریداری", "خریدن", "بخریم", "خریدم", "خرید کنم", "buy", "خریدار"],
    "sell": ["فروش", "بفروشم", "می‌فروشم", "میفروشم", "فروختن", "بفروشیم", "فروشنده", "فروختم", "sell", "فروش کنم"]
}

CURRENCY_NAMES = {
    "IRR": "تومان",
    "AED": "درهم",
    "USDT": "تتر"
}

EXCHANGE_RATE_ENDPOINTS = {
    ("USDT", "AED"): "https://api.abanprime.com/futures/market/USDT-AED",
    ("USDT", "IRR"): "https://api.abanprime.com/futures/market/USDT-IRT",
    ("AED", "IRR"): "https://api.abanprime.com/futures/market/AED-IRT"
}

# Pydantic models for query classification
class QueryType(BaseModel):
    """Classify a user query into specific types for routing."""
    
    query_type: Literal["faq", "exchange_rate", "transaction", "other"] = Field(
        ...,
        description="Classify the user query into one of these categories: 'faq' for general information, 'exchange_rate' for currency rate inquiries, 'transaction' for buying/selling requests, 'other' for queries that don't fit these categories"
    )
    
    source_currency: str = Field(
        default="",
        description="If this is an exchange_rate or transaction query, identify the source currency (IRR, AED, USDT, etc.). Leave empty if not applicable."
    )
    
    target_currency: str = Field(
        default="",
        description="If this is an exchange_rate or transaction query, identify the target currency (IRR, AED, USDT, etc.). Leave empty if not applicable."
    )
    
    transaction_type: Literal["buy", "sell", "unknown"] = Field(
        default="unknown",
        description="If this is a transaction query, identify if it's a buy or sell request. Use 'unknown' if not clear or not applicable."
    )
    
    amount: str = Field(
        default="",
        description="Extract any mentioned currency amount. Leave empty if not specified."
    )

# ===================================
# Embedding and Retrieval Functions
# ===================================

def get_embedding(text: str) -> list:
    """Generate embedding using the fine-tuned model."""
    emb = ft_model.encode(text)
    return emb.tolist()

def query_index(query: str) -> str:
    """
    Queries the Pinecone index with the query's embedding and 
    returns concatenated context from the top matching chunks.
    """
    query_emb = get_embedding(query)
    results = index.query(
        namespace="faq-namespace",
        vector=query_emb,
        top_k=3,
        include_values=False,
        include_metadata=True
    )
    print("RAG Results:", results)
    
    # Extract text from metadata
    retrieved_texts = [match["metadata"]["text"] for match in results.get("matches", [])]
    return "\n\n".join(retrieved_texts)

def improved_query_index(query: str) -> str:
    """
    Enhanced retrieval function with query reformulation and hybrid search.
    """
    try:
        # Step 1: Reformulate the query for better retrieval
        reformulated_query = reformulate_query(query)
        
        # Step 2: Generate embeddings for reformulated query
        query_emb = get_embedding(reformulated_query)
        
        # Step 3: Perform vector search
        results = index.query(
            namespace="faq-namespace",
            vector=query_emb,
            top_k=5,  # Increased from 3 to 5 for better coverage
            include_values=False,
            include_metadata=True
        )
        
        # Step 4: Rerank results (simple implementation)
        # This could be enhanced with a proper reranking model
        reranked_results = simple_rerank(results, query)
        
        # Extract and join retrieved texts
        retrieved_texts = [match["metadata"]["text"] for match in reranked_results]
        return "\n\n".join(retrieved_texts)
    except Exception as e:
        print(f"Error in improved_query_index: {e}")
        # Fallback to the original query method if there's an error
        return query_index(query)

def reformulate_query(query: str) -> str:
    """Use LLM to reformulate the query for better retrieval."""
    prompt = (
        "Reformulate this user query to optimize it for retrieval from a FAQ database "
        "about currency exchange services, while preserving the original intent.\n\n"
        f"Original query: {query}\n"
        "Reformulated query:"
    )
    
    response = client.chat.completions.create(
        model="gpt-4o",  # Use a smaller model for efficiency
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=100
    )
    
    return response.choices[0].message.content.strip()

def simple_rerank(results: dict, original_query: str) -> List[dict]:
    """
    Simple reranking function based on keyword matching.
    In a production system, this would be replaced with a proper reranking model.
    """
    matches = results.get("matches", [])
    if not matches:
        return []
    
    # Extract important keywords from the query
    query_terms = set(original_query.lower().split())
    
    # Score each result based on keyword overlap
    scored_matches = []
    for match in matches:
        text = match["metadata"].get("text", "").lower()
        score = sum(1 for term in query_terms if term in text)
        # Store score and index to avoid comparing ScoredVector objects
        scored_matches.append((score, len(scored_matches), match))
    
    # Sort by score (descending) and then by original index as a tiebreaker
    # This avoids comparing ScoredVector objects directly
    scored_matches.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    return [match for _, _, match in scored_matches]

# ===================================
# Query Classification Functions
# ===================================

def classify_query(query: str) -> QueryType:
    """
    Classify the query using OpenAI function calling.
    """
    system_prompt = """
    You are an expert at analyzing financial queries in both Persian (Farsi).
    Your task is to classify user questions about currency exchange, rates, and transactions.

    Currency information:
    - IRR/IRT: Iranian Rial/Toman (تومان، ریال، تومن)
    - AED: UAE Dirham (درهم، درهم امارات)
    - USDT: Tether/USD stablecoin (تتر، دلار)

    For transaction queries, determine if it's a buy or sell request.
    For rate queries, identify both the source and target currencies.
    Extract any specific amounts mentioned (only for transaction or exchange rate queries).

    For general questions or FAQ queries, leave the currency, transaction type, and amount fields empty.
    """
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "classify_query",
                "description": "Classify a user query about currency exchange",
                "parameters": QueryType.schema()
            }
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "classify_query"}},
            temperature=0.1
        )
        
        # Extract the function call result
        function_call = response.choices[0].message.tool_calls[0]
        classification_data = json.loads(function_call.function.arguments)
        
        # Ensure all expected fields are present in the response
        # Set default values if keys are missing
        required_fields = {
            "query_type": "faq",
            "source_currency": "",
            "target_currency": "",
            "transaction_type": "unknown",
            "amount": ""
        }
        
        for key, default_value in required_fields.items():
            if key not in classification_data:
                classification_data[key] = default_value
        
        # Only for exchange rate and transaction queries:
        # If a currency is detected but amount is missing, try to extract it with regex
        if classification_data["query_type"] in ["transaction", "exchange_rate"]:
            # Only try to extract the amount if it's relevant for the query type
            if not classification_data["amount"]:
                amount = extract_amount(query)
                if amount:
                    classification_data["amount"] = amount
        
        return QueryType(**classification_data)
    
    except Exception as e:
        print(f"Error in classify_query: {e}")
        # Return a default classification if there's an error
        return QueryType(
            query_type="faq",
            source_currency="",
            target_currency="",
            transaction_type="unknown",
            amount=""
        )

def extract_amount(text: str) -> Optional[str]:
    """Extract numerical amounts from text using regex."""
    # Match Persian and English digits with potential thousand separators
    pattern = r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)'
    matches = re.findall(pattern, text)
    
    if matches:
        # Return the first match (could be enhanced to identify the most relevant amount)
        return matches[0]
    return None

def detect_currencies_in_text(text: str) -> Dict[str, float]:
    """
    Detect currency mentions and their likelihood in the text.
    Returns a dictionary mapping currency codes to confidence scores.
    """
    text_lower = text.lower()
    currency_scores = {}
    
    for currency, keywords in CURRENCY_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword.lower() in text_lower:
                score += 1
        if score > 0:
            currency_scores[currency] = score
    
    # Normalize scores if any currencies were found
    if currency_scores:
        total_score = sum(currency_scores.values())
        for currency in currency_scores:
            currency_scores[currency] /= total_score
    
    return currency_scores

def detect_transaction_type(text: str) -> str:
    """
    Detect if the text is about buying or selling.
    Returns 'buy', 'sell', or 'unknown'.
    """
    text_lower = text.lower()
    
    buy_score = 0
    for keyword in TRANSACTION_KEYWORDS["buy"]:
        if keyword.lower() in text_lower:
            buy_score += 1
    
    sell_score = 0
    for keyword in TRANSACTION_KEYWORDS["sell"]:
        if keyword.lower() in text_lower:
            sell_score += 1
    
    if buy_score > sell_score:
        return "buy"
    elif sell_score > buy_score:
        return "sell"
    else:
        return "unknown"

# ===================================
# Exchange Rate Functions
# ===================================

def get_exchange_rate(source_currency: str, target_currency: str) -> Optional[Dict[str, float]]:
    """Get the current exchange rate between two currencies."""
    # Determine the correct API endpoint
    currency_pair = (source_currency, target_currency)
    invert_rate = False
    
    if currency_pair in EXCHANGE_RATE_ENDPOINTS:
        endpoint = EXCHANGE_RATE_ENDPOINTS[currency_pair]
    elif (target_currency, source_currency) in EXCHANGE_RATE_ENDPOINTS:
        endpoint = EXCHANGE_RATE_ENDPOINTS[(target_currency, source_currency)]
        # Will need to invert the rate later
        invert_rate = True
    else:
        # Handle unsupported pairs
        print(f"Unsupported currency pair: {source_currency}-{target_currency}")
        return None
    
    try:
        # Call the API
        response = requests.get(endpoint)
        if response.status_code == 200:
            json_data = response.json()
            best_sell_price = float(json_data["sellOrderBook"]["bestPrice"])
            best_buy_price = float(json_data["buyOrderBook"]["bestPrice"])
            
            if invert_rate:
                # Invert for reversed currency pairs
                return {
                    "buy_rate": 1 / best_sell_price,  # Inverted because buying X means selling Y
                    "sell_rate": 1 / best_buy_price   # Inverted because selling X means buying Y
                }
            else:
                return {
                    "buy_rate": best_buy_price,
                    "sell_rate": best_sell_price
                }
        else:
            print(f"API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching exchange rate: {e}")
        return None

def generate_rate_response(source_currency: str, target_currency: str, 
                           rate_data: Optional[Dict[str, float]], query: str, 
                           conversation_history: str) -> str:
    """Generate a natural language response about exchange rates."""
    if not rate_data:
        # Handle case where rate data couldn't be retrieved
        prompt = (
            "You are a helpful financial assistant for AbanPrime (آبان پرایم) currency exchange. "
            f"The user asked about the exchange rate between {source_currency} and {target_currency}, "
            "but we couldn't retrieve the current rates. "
            "Apologize and offer to help them with other currency pairs we support (USDT-AED, USDT-IRR, AED-IRR).\n\n"
            "Conversation so far:\n"
            f"{conversation_history}\n"
            f"User: {query}\n"
            "Assistant:"
        )
    else:
        # Use the retrieved rate data
        source_name = CURRENCY_NAMES.get(source_currency, source_currency)
        target_name = CURRENCY_NAMES.get(target_currency, target_currency)
        
        context = f"""
        Current exchange rates between {source_name} ({source_currency}) and {target_name} ({target_currency}):
        - 1 {source_currency} can be bought for {rate_data['buy_rate']} {target_currency}
        - 1 {source_currency} can be sold for {rate_data['sell_rate']} {target_currency}
        
        Currency names in Persian:
        - IRR: تومان
        - AED: درهم
        - USDT: تتر
        """
        
        prompt = (
            "You are a helpful financial assistant for AbanPrime (آبان پرایم) currency exchange. "
            "Based on the following exchange rate information, answer the user's question in a conversational manner. "
            "Provide the exact rates and suggest if they want to proceed with a transaction.\n\n"
            f"Context: {context}\n\n"
            "Conversation so far:\n"
            f"{conversation_history}\n"
            f"User: {query}\n"
            "Assistant:"
        )
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# ===================================
# Transaction Functions
# ===================================

def handle_transaction(query_info: QueryType, conversation_history: str) -> str:
    """Handle a transaction request with the extracted information."""
    # Extract info from the query classification
    transaction_type = query_info.transaction_type
    source_currency = query_info.source_currency
    target_currency = query_info.target_currency
    amount = query_info.amount
    
    # Check if we have all the necessary information
    missing_info = []
    if transaction_type == "unknown":
        missing_info.append("whether you want to buy or sell")
    if not source_currency:
        missing_info.append("which currency you're starting with")
    if not target_currency:
        missing_info.append("which currency you want to receive")
    if not amount:
        missing_info.append("how much you want to exchange")
    
    # If information is missing, ask follow-up questions
    if missing_info:
        prompt = (
            "You are a helpful financial assistant for AbanPrime currency exchange. "
            "The user wants to make a transaction but some information is missing. "
            f"Missing information: {', '.join(missing_info)}.\n\n"
            "Ask follow-up questions in a friendly, conversational manner to get the missing information. "
            "Be specific about what you need to know. Ask the questions in Persian (Farsi) and English.\n\n"
            "Conversation so far:\n"
            f"{conversation_history}\n"
            f"User's last message: {query_info.source_currency}, {query_info.target_currency}, {query_info.transaction_type}, {query_info.amount}\n"
            "Assistant:"
        )
    else:
        # If we have all information, get the exchange rate
        rate_data = get_exchange_rate(source_currency, target_currency)
        
        if rate_data:
            # Calculate transaction details
            if transaction_type == "buy":
                rate = rate_data["buy_rate"]
                total_amount = float(amount.replace(',', '')) * rate
            else:  # sell
                rate = rate_data["sell_rate"]
                total_amount = float(amount.replace(',', '')) * rate
            
            # Format amounts with commas for thousands
            formatted_amount = "{:,.2f}".format(float(amount.replace(',', '')))
            formatted_total = "{:,.2f}".format(total_amount)
            
            source_name = CURRENCY_NAMES.get(source_currency, source_currency)
            target_name = CURRENCY_NAMES.get(target_currency, target_currency)
            
            prompt = (
                "You are a helpful financial assistant for AbanPrime (آبان پرایم) currency exchange. "
                "Summarize the transaction details below and ask if the user wants to proceed. "
                "Use both Persian (Farsi) and English in your response.\n\n"
                f"Transaction Type: {transaction_type.upper()}\n"
                f"Amount: {formatted_amount} {source_currency} ({source_name})\n"
                f"Rate: 1 {source_currency} = {rate} {target_currency}\n"
                f"Total: {formatted_total} {target_currency} ({target_name})\n\n"
                "Explain the process for completing this transaction step by step (1, 2, 3...).\n\n"
                "Conversation so far:\n"
                f"{conversation_history}\n"
                "Assistant:"
            )
        else:
            prompt = (
                "You are a helpful financial assistant for AbanPrime currency exchange. "
                f"The user wants to {transaction_type} {amount} {source_currency} for {target_currency}, "
                "but we couldn't retrieve the current exchange rate. "
                "Apologize and offer alternatives or suggest trying again later. "
                "Respond in both Persian (Farsi) and English.\n\n"
                "Conversation so far:\n"
                f"{conversation_history}\n"
                "Assistant:"
            )
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# ===================================
# FAQ Response Functions
# ===================================

def generate_faq_response(query: str, context: str, conversation_history: str) -> str:
    """
    Builds a prompt with the retrieved context and generates a conversational
    answer using GPT-4o.
    """
    prompt = (
        "You are a friendly, helpful, and natural conversational AI assistant. "
        "You work for a currency exchange company named AbanPrime (آبان پرایم in Persian). "
        "Based on the following context extracted from your indexed documents, "
        "answer the user's question in a supportive way. "
        "If explaining a procedure, create a step-by-step workflow (1, 2, 3...). "
        "If appropriate, ask follow-up clarifying questions like 'How much USDT would you like to buy?' "
        "or 'How will you pay? In which currency?'.\n\n"
        "Context from the vector database:\n"
        f"{context}\n\n"
        "Conversation so far:\n"
        f"{conversation_history}\n"
        f"User: {query}\n"
        "Assistant:"
    )

    chat_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return chat_response.choices[0].message.content

# ===================================
# Main Orchestration Function
# ===================================

def process_query(user_query: str, conversation_history: str) -> str:
    """Process the user query and route to the appropriate pipeline."""
    # Classify the query
    classification = classify_query(user_query)
    query_type = classification.query_type
    
    print(f"Query classified as: {query_type}")
    print(f"Classification details: {classification}")
    
    # Route to the appropriate pipeline
    if query_type == "faq":
        # Use existing RAG pipeline with improvements
        context = improved_query_index(user_query)
        response = generate_faq_response(user_query, context, conversation_history)
    
    elif query_type == "exchange_rate":
        # Use exchange rate pipeline
        source = classification.source_currency
        target = classification.target_currency
        
        # If currencies are not identified, try to extract them
        if not source or not target:
            currency_scores = detect_currencies_in_text(user_query)
            
            # Get the top 2 currencies by score
            top_currencies = sorted(currency_scores.items(), key=lambda x: x[1], reverse=True)
            
            if len(top_currencies) >= 2:
                source, _ = top_currencies[0]
                target, _ = top_currencies[1]
            elif len(top_currencies) == 1:
                source, _ = top_currencies[0]
                # Try to guess the other currency - common pairs
                if source == "USDT":
                    target = "IRR"  # Default target for USDT is IRR
                elif source == "IRR":
                    target = "USDT"  # Default target for IRR is USDT
                else:
                    target = "USDT"  # Default fallback
        
        # Get exchange rate and generate response
        rate_data = get_exchange_rate(source, target)
        response = generate_rate_response(source, target, rate_data, user_query, conversation_history)
    
    elif query_type == "transaction":
        # Use transaction pipeline
        response = handle_transaction(classification, conversation_history)
    
    else:  # "other"
        # Fallback to general assistant
        prompt = (
            "You are a helpful assistant for AbanPrime currency exchange. "
            "The user has asked a question that doesn't specifically relate to FAQs, exchange rates, or transactions. "
            "Please provide a helpful response and guide them toward services you can provide.\n\n"
            "Conversation so far:\n"
            f"{conversation_history}\n"
            f"User: {user_query}\n"
            "Assistant:"
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        response = response.choices[0].message.content
    
    return response

# ===================================
# Streamlit Interface with Memory
# ===================================

def main():
    """Main Streamlit app with memory pipeline integration."""
    st.title("🔄 AbanPrime Chat Assistant")
    st.subheader("Your Currency Exchange Expert")
    
    # Initialize memory in session state
    if "memory" not in st.session_state:
        st.session_state.memory = initialize_memory()
    
    # Create two columns: one for the conversation and one for transaction details
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create a container for chat history with fixed height and scrolling
        chat_container = st.container()
        with chat_container:
            st.markdown("""
            <style>
            .chat-container {
                height: 400px;
                overflow-y: auto;
                border: 1px solid #ddd;
                padding: 15px;
                border-radius: 5px;
            }
            </style>
            <div class="chat-container" id="chat-container">
            """, unsafe_allow_html=True)
            
            # Display the conversation history
            for msg in st.session_state.memory.messages:
                if msg.role == "user":
                    st.markdown(f"**You:** {msg.content}")
                else:
                    st.markdown(f"**Assistant:** {msg.content}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Text input for the user's query
        user_query = st.text_input("Ask a question:", key="user_input")
        
        send_col, clear_col = st.columns([5, 1])
        with send_col:
            send_button = st.button("Send", use_container_width=True)
        with clear_col:
            if st.button("Clear", use_container_width=True):
                st.session_state.memory = initialize_memory()
                st.rerun()
        
        if send_button and user_query:
            with st.spinner("Generating response..."):
                # Process the query using our memory pipeline
                response, updated_memory, transaction_status = process_conversation_with_memory(
                    user_query, 
                    st.session_state.memory,
                    client,
                    process_query,
                    get_exchange_rate
                )
                
                # Update the memory in session state
                st.session_state.memory = updated_memory
                
                # Rerun to update the display - the input will be cleared
                # because we're not preserving its value between reruns
                st.rerun()
    
    with col2:
        # Display transaction information if available
        if st.session_state.memory.transaction_status == "pending":
            st.markdown("### 💼 Pending Transaction")
            
            details = st.session_state.memory.transaction_details
            
            st.markdown(f"**Type:** {details.get('transaction_type', 'N/A').capitalize()}")
            st.markdown(f"**From:** {details.get('amount', 'N/A')} {details.get('source_currency', 'N/A')}")
            st.markdown(f"**To:** {details.get('target_currency', 'N/A')}")
            
            if "rate" in details:
                st.markdown(f"**Rate:** {details.get('rate', 'N/A')}")
            
            if "formatted_total" in details:
                st.markdown(f"**Total:** {details.get('formatted_total', 'N/A')} {details.get('target_currency', 'N/A')}")
            
            # Confirm/Cancel buttons
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ Confirm", use_container_width=True):
                    with st.spinner("Processing transaction..."):
                        # Process confirmation as a user message
                        response, updated_memory, transaction_status = process_conversation_with_memory(
                            "I confirm this transaction",
                            st.session_state.memory,
                            client,
                            process_query,
                            get_exchange_rate
                        )
                        st.session_state.memory = updated_memory
                        st.rerun()
            
            with col_b:
                if st.button("❌ Cancel", use_container_width=True):
                    with st.spinner("Cancelling transaction..."):
                        # Process cancellation as a user message
                        response, updated_memory, transaction_status = process_conversation_with_memory(
                            "Cancel this transaction",
                            st.session_state.memory,
                            client,
                            process_query,
                            get_exchange_rate
                        )
                        st.session_state.memory = updated_memory
                        st.rerun()
        else:
            # Display current exchange rates
            st.markdown("### 💱 Current Exchange Rates")
            
            try:
                usdt_aed = get_exchange_rate("USDT", "AED")
                usdt_irr = get_exchange_rate("USDT", "IRR")
                aed_irr = get_exchange_rate("AED", "IRR")
                
                if usdt_aed:
                    st.markdown(f"**USDT/AED:**")
                    st.markdown(f"- Buy: {usdt_aed['buy_rate']:.4f}")
                    st.markdown(f"- Sell: {usdt_aed['sell_rate']:.4f}")
                
                if usdt_irr:
                    st.markdown(f"**USDT/IRR:**")
                    st.markdown(f"- Buy: {usdt_irr['buy_rate']:.0f}")
                    st.markdown(f"- Sell: {usdt_irr['sell_rate']:.0f}")
                
                if aed_irr:
                    st.markdown(f"**AED/IRR:**")
                    st.markdown(f"- Buy: {aed_irr['buy_rate']:.0f}")
                    st.markdown(f"- Sell: {aed_irr['sell_rate']:.0f}")
            except Exception as e:
                st.markdown("*Unable to fetch current rates*")
        
        # Display conversation summary if available
        if st.session_state.memory.summary:
            st.markdown("### 📝 Conversation Summary")
            st.info(st.session_state.memory.summary)
        
        # Show follow-up suggestions if available
        if st.session_state.memory.follow_up_questions:
            st.markdown("### 💬 Suggested Questions")
            for question in st.session_state.memory.follow_up_questions:
                question_text = question[:30] + "..." if len(question) > 30 else question
                if st.button(question_text, key=f"q_{hash(question)}"):
                    # Process the suggested question as a user message
                    response, updated_memory, transaction_status = process_conversation_with_memory(
                        question,
                        st.session_state.memory,
                        client,
                        process_query,
                        get_exchange_rate
                    )
                    st.session_state.memory = updated_memory
                    st.rerun()

if __name__ == "__main__":
    main()