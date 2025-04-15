import time
import streamlit as st
import requests
import json
from datetime import datetime
from typing import List, Dict, Optional, Union, Any
from pydantic import BaseModel, Field
import uuid

# ===================================
# Conversation Memory Models
# ===================================

class Message(BaseModel):
    """Model for a conversation message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = Field(default_factory=time.time)

class EntityValue(BaseModel):
    """Model for an extracted entity value with confidence."""
    value: str
    confidence: float = 1.0
    extracted_from: str = ""  # The message this was extracted from

class ConversationMemory(BaseModel):
    """Model for the entire conversation memory."""
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = []
    entities: Dict[str, EntityValue] = {}
    summary: str = ""
    transaction_status: str = "none"  # none, pending, confirmed, completed, cancelled
    transaction_details: Dict[str, Any] = {}
    follow_up_questions: List[str] = []

# ===================================
# Conversation Memory Functions
# ===================================

def initialize_memory() -> ConversationMemory:
    """Initialize a new conversation memory."""
    return ConversationMemory()

def add_message_to_memory(memory: ConversationMemory, role: str, content: str) -> ConversationMemory:
    """Add a new message to the conversation memory."""
    memory.messages.append(Message(role=role, content=content))
    return memory

def extract_entities(memory: ConversationMemory, message: str, 
                    client: Any) -> ConversationMemory:
    """
    Extract relevant entities from the latest message and update memory.
    Uses the OpenAI API to extract structured information.
    """
    # Define the entity extraction schema
    entities_schema = {
        "type": "object",
        "properties": {
            "currencies": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "currency": {"type": "string"},
                        "confidence": {"type": "number"}
                    }
                }
            },
            "amounts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "amount": {"type": "string"},
                        "currency": {"type": "string"},
                        "confidence": {"type": "number"}
                    }
                }
            },
            "transaction_type": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["buy", "sell", "exchange", "none"]},
                    "confidence": {"type": "number"}
                }
            },
            "payment_method": {
                "type": "object",
                "properties": {
                    "method": {"type": "string"},
                    "confidence": {"type": "number"}
                }
            }
        }
    }

    system_prompt = """
    You are an expert at extracting information from conversations about currency exchange.
    Extract entities like currencies, amounts, transaction types, and payment methods.
    Only return values that are explicitly mentioned in the message.
    If information is not present, don't guess - leave it empty or set confidence to 0.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract information from this message: '{message}'"}
            ],
            response_format={"type": "json_object", "schema": entities_schema},
            temperature=0.1
        )
        
        extracted_data = json.loads(response.choices[0].message.content)
        
        # Update memory with extracted currencies
        for currency_item in extracted_data.get("currencies", []):
            if currency_item.get("confidence", 0) > 0.7:  # Only add high-confidence entities
                currency = currency_item["currency"].upper()  # Normalize to uppercase
                memory.entities[f"currency_{len(memory.entities) + 1}"] = EntityValue(
                    value=currency,
                    confidence=currency_item["confidence"],
                    extracted_from=message
                )
        
        # Update with extracted amounts
        for amount_item in extracted_data.get("amounts", []):
            if amount_item.get("confidence", 0) > 0.7:
                amount = amount_item["amount"]
                currency = amount_item.get("currency", "").upper()
                # Create a key that includes the currency if available
                key = f"amount_{currency}" if currency else f"amount_{len(memory.entities) + 1}"
                memory.entities[key] = EntityValue(
                    value=amount,
                    confidence=amount_item["confidence"],
                    extracted_from=message
                )
        
        # Update transaction type
        txn_type = extracted_data.get("transaction_type", {})
        if txn_type.get("confidence", 0) > 0.7 and txn_type.get("type") != "none":
            memory.entities["transaction_type"] = EntityValue(
                value=txn_type["type"],
                confidence=txn_type["confidence"],
                extracted_from=message
            )
        
        # Update payment method
        payment = extracted_data.get("payment_method", {})
        if payment.get("confidence", 0) > 0.7:
            memory.entities["payment_method"] = EntityValue(
                value=payment["method"],
                confidence=payment["confidence"],
                extracted_from=message
            )
    
    except Exception as e:
        print(f"Error extracting entities: {e}")
    
    return memory

def generate_conversation_summary(memory: ConversationMemory, client: Any) -> str:
    """
    Generate a summary of the conversation focused on transaction intent.
    """
    if len(memory.messages) < 2:
        return ""
    
    # Prepare conversation history
    conversation = "\n".join([
        f"{msg.role.capitalize()}: {msg.content}" for msg in memory.messages[-10:]  # Last 10 messages
    ])
    
    system_prompt = """
    You are an expert at summarizing conversations about currency exchange.
    Create a concise summary of the user's intent, focusing on:
    1. What transaction they want to make (buy, sell, exchange)
    2. Which currencies are involved
    3. The amounts they mentioned
    4. Any specific requirements or concerns
    
    Only include information that was explicitly stated in the conversation.
    Format your response as a short paragraph.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize this conversation:\n\n{conversation}"}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        summary = response.choices[0].message.content.strip()
        return summary
    
    except Exception as e:
        print(f"Error generating summary: {e}")
        return ""

def identify_missing_information(memory: ConversationMemory, client: Any) -> List[str]:
    """
    Identify what information is still needed to complete a transaction.
    Returns a list of questions to ask the user.
    """
    # First check if we have a transaction type
    has_transaction_intent = "transaction_type" in memory.entities
    
    if not has_transaction_intent:
        # No transaction intent detected yet
        return []
    
    # Prepare the entity information we've gathered
    entities_info = "\n".join([
        f"{key}: {entity.value} (confidence: {entity.confidence})"
        for key, entity in memory.entities.items()
    ])
    
    system_prompt = """
    You are an expert at identifying missing information needed for currency exchange transactions.
    Based on the conversation and extracted entities, determine what critical information is still missing.
    
    For a complete transaction, we typically need:
    1. Transaction type (buy, sell, exchange)
    2. Source currency
    3. Target currency
    4. Amount
    5. Payment method
    
    Generate 1-2 specific follow-up questions to obtain the missing information.
    Format your response as a JSON list of strings, each containing one question.
    Questions should be in both Persian (Farsi) and English.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Conversation entities:\n{entities_info}\n\nWhat information is still missing?"}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        questions = json.loads(response.choices[0].message.content)
        return questions.get("questions", [])
    
    except Exception as e:
        print(f"Error identifying missing information: {e}")
        return []

def prepare_transaction_confirmation(memory: ConversationMemory, client: Any) -> Dict[str, Any]:
    """
    Prepare a transaction confirmation summary with all collected information.
    """
    # Extract relevant entities for the transaction
    transaction_type = memory.entities.get("transaction_type", EntityValue(value="unknown", confidence=0)).value
    
    # Find source and target currencies
    source_currency = None
    target_currency = None
    
    # Look for currency entities
    currency_entities = {k: v for k, v in memory.entities.items() if k.startswith("currency_")}
    if len(currency_entities) >= 2:
        # Take the two highest confidence currencies
        sorted_currencies = sorted(currency_entities.items(), key=lambda x: x[1].confidence, reverse=True)
        source_currency = sorted_currencies[0][1].value
        target_currency = sorted_currencies[1][1].value
    elif len(currency_entities) == 1:
        source_currency = list(currency_entities.values())[0].value
        # If we only have one currency, try to determine the other based on transaction type
        if transaction_type == "buy":
            # When buying, the target is what we're getting
            target_currency = source_currency
            source_currency = "IRR"  # Default assumption
        elif transaction_type == "sell":
            # When selling, the source is what we're giving up
            target_currency = "IRR"  # Default assumption
    
    # Find amount
    amount = None
    amount_keys = [k for k in memory.entities.keys() if k.startswith("amount_")]
    if amount_keys:
        amount = memory.entities[amount_keys[0]].value
    
    # Prepare transaction details
    transaction_details = {
        "transaction_type": transaction_type,
        "source_currency": source_currency,
        "target_currency": target_currency,
        "amount": amount,
        "payment_method": memory.entities.get("payment_method", EntityValue(value="", confidence=0)).value,
        "timestamp": datetime.now().isoformat()
    }
    
    # If we have enough information, calculate the estimated total
    if transaction_type != "unknown" and source_currency and target_currency and amount:
        # Here you would call your exchange rate API, but we'll use placeholder values
        transaction_details["estimated_total"] = f"[Based on current exchange rates]"
        transaction_details["has_minimum_info"] = True
    else:
        transaction_details["has_minimum_info"] = False
    
    return transaction_details

def generate_follow_up_questions(memory: ConversationMemory, client: Any) -> List[str]:
    """
    Generate follow-up questions based on the conversation flow.
    This tries to move the conversation forward naturally.
    """
    # If we already have transaction details and enough info, suggest confirmation
    if (memory.transaction_status == "pending" and 
        memory.transaction_details.get("has_minimum_info", False)):
        return ["Would you like to confirm this transaction?"]
    
    # Otherwise, look for missing information
    missing_info_questions = identify_missing_information(memory, client)
    if missing_info_questions:
        return missing_info_questions
    
    # If no specific missing info, but we're in a conversation flow, generate general follow-ups
    if len(memory.messages) >= 2:
        conversation = "\n".join([
            f"{msg.role.capitalize()}: {msg.content}" for msg in memory.messages[-5:]
        ])
        
        system_prompt = """
        You are an expert conversational assistant for a currency exchange service.
        Based on the conversation so far, generate one natural follow-up question that:
        1. Keeps the conversation flowing naturally
        2. Helps understand what the user wants to do
        3. Moves toward completing a transaction if appropriate
        
        Your question should be in both Persian (Farsi) and English.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Current conversation:\n\n{conversation}"}
                ],
                temperature=0.4,
                max_tokens=100
            )
            
            return [response.choices[0].message.content.strip()]
            
        except Exception as e:
            print(f"Error generating follow-up questions: {e}")
    
    return ["Is there anything else I can help you with today?"]

def create_transaction_summary(memory: ConversationMemory, client: Any) -> str:
    """
    Create a final transaction summary for the user.
    This is shown after the user confirms they want to proceed.
    """
    details = memory.transaction_details
    
    transaction_info = {
        "transaction_type": details.get("transaction_type", "unknown"),
        "source_currency": details.get("source_currency", ""),
        "target_currency": details.get("target_currency", ""),
        "amount": details.get("amount", ""),
        "estimated_total": details.get("estimated_total", ""),
        "payment_method": details.get("payment_method", ""),
        "timestamp": details.get("timestamp", datetime.now().isoformat())
    }
    
    system_prompt = """
    You are an expert financial assistant for AbanPrime currency exchange.
    Create a clear, detailed transaction summary based on the information provided.
    
    Include:
    1. A confirmation ID number (make one up)
    2. Transaction type and currencies involved
    3. Amount and estimated total
    4. Next steps the user needs to take to complete the transaction
    5. Customer support contact information
    
    Format this attractively and professionally in both Persian (Farsi) and English.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Using the larger model for a polished final output
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Create a transaction summary with these details:\n\n{json.dumps(transaction_info, indent=2)}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error creating transaction summary: {e}")
        return "There was an error creating your transaction summary. Please contact customer support."

# ===================================
# Integrated Pipeline Function
# ===================================

def process_conversation_with_memory(
        user_query: str, 
        memory: ConversationMemory, 
        client: Any,
        process_query_func: callable,
        get_exchange_rate_func: callable
    ) -> tuple:
    """
    Process the user query with conversation memory tracking.
    Returns a tuple of (response, updated_memory, transaction_status).
    """
    # Add user message to memory
    memory = add_message_to_memory(memory, "user", user_query)
    
    # Extract entities from the user's message
    memory = extract_entities(memory, user_query, client)
    
    # Check for transaction confirmation intent
    confirmation_phrases = [
        "confirm", "proceed", "go ahead", "do it", "تایید", "تأیید", "انجام بده", "قبول"
    ]
    
    cancellation_phrases = [
        "cancel", "stop", "don't proceed", "لغو", "انصراف", "متوقف", "نمی‌خواهم"
    ]
    
    # Check if this is a confirmation of a pending transaction
    if (memory.transaction_status == "pending" and 
        any(phrase in user_query.lower() for phrase in confirmation_phrases)):
        # User is confirming the transaction
        memory.transaction_status = "confirmed"
        
        # Create a detailed transaction summary
        summary = create_transaction_summary(memory, client)
        
        # Add assistant response to memory
        memory = add_message_to_memory(memory, "assistant", summary)
        
        return summary, memory, "confirmed"
    
    # Check if this is a cancellation of a pending transaction
    elif (memory.transaction_status == "pending" and 
          any(phrase in user_query.lower() for phrase in cancellation_phrases)):
        # User is cancelling the transaction
        memory.transaction_status = "cancelled"
        
        # Create cancellation message
        cancel_message = (
            "I've cancelled this transaction. Is there something else you'd like help with?\n\n"
            "من این تراکنش را لغو کردم. آیا کمک دیگری نیاز دارید؟"
        )
        
        # Add assistant response to memory
        memory = add_message_to_memory(memory, "assistant", cancel_message)
        
        return cancel_message, memory, "cancelled"
    
    # Normal flow - process through the query pipeline
    response = process_query_func(user_query, "\n".join([
        f"{msg.role.capitalize()}: {msg.content}" for msg in memory.messages
    ]))
    
    # Add assistant response to memory
    memory = add_message_to_memory(memory, "assistant", response)
    
    # Update conversation summary
    memory.summary = generate_conversation_summary(memory, client)
    
    # Check if we've collected enough information for a transaction
    transaction_entities = prepare_transaction_confirmation(memory, client)
    
    # If we have minimum info for a transaction and it's not pending already
    if (transaction_entities.get("has_minimum_info", False) and 
        memory.transaction_status != "pending" and
        "transaction_type" in memory.entities):
        
        # Update transaction details
        memory.transaction_details = transaction_entities
        memory.transaction_status = "pending"
        
        # Get actual exchange rate for the transaction
        if (transaction_entities.get("source_currency") and 
            transaction_entities.get("target_currency")):
            
            rate_data = get_exchange_rate_func(
                transaction_entities["source_currency"], 
                transaction_entities["target_currency"]
            )
            
            if rate_data:
                # Calculate actual total
                try:
                    amount = float(transaction_entities["amount"].replace(',', ''))
                    
                    if transaction_entities["transaction_type"] == "buy":
                        rate = rate_data["buy_rate"]
                    else:  # sell or exchange
                        rate = rate_data["sell_rate"]
                    
                    total = amount * rate
                    
                    # Update transaction details with actual rate and total
                    memory.transaction_details["rate"] = rate
                    memory.transaction_details["total"] = total
                    memory.transaction_details["formatted_total"] = f"{total:,.2f}"
                    
                except Exception as e:
                    print(f"Error calculating transaction total: {e}")
        
        # Generate follow-up questions to confirm transaction
        memory.follow_up_questions = ["Would you like to confirm this transaction?"]
    else:
        # Generate appropriate follow-up questions based on context
        memory.follow_up_questions = generate_follow_up_questions(memory, client)
    
    return response, memory, memory.transaction_status

# ===================================
# Streamlit Interface with Memory
# ===================================

def initialize_streamlit_memory():
    """Initialize the memory in Streamlit session state."""
    if "memory" not in st.session_state:
        st.session_state.memory = initialize_memory()

def main_with_memory(process_query_func, get_exchange_rate_func, client):
    """Main Streamlit app with memory pipeline integration."""
    st.title("AbanPrime Chat Assistant")
    st.subheader("Your Currency Exchange Expert")
    
    # Initialize memory
    initialize_streamlit_memory()
    
    # Create two columns: one for the conversation and one for transaction details
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display the conversation history
        for msg in st.session_state.memory.messages:
            if msg.role == "user":
                st.markdown(f"**You:** {msg.content}")
            else:
                st.markdown(f"**Assistant:** {msg.content}")
        
        # Text input for the user's query
        user_query = st.text_input("Ask a question:")
        
        if st.button("Send") and user_query:
            with st.spinner("Generating response..."):
                # Process the query using our memory pipeline
                response, updated_memory, transaction_status = process_conversation_with_memory(
                    user_query, 
                    st.session_state.memory,
                    client,
                    process_query_func,
                    get_exchange_rate_func
                )
                
                # Update the memory in session state
                st.session_state.memory = updated_memory
                
                # Rerun to update the display
                st.experimental_rerun()
    
    with col2:
        # Display transaction information if available
        if st.session_state.memory.transaction_status == "pending":
            st.subheader("Pending Transaction")
            
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
                if st.button("Confirm"):
                    with st.spinner("Processing transaction..."):
                        # Process confirmation as a user message
                        response, updated_memory, transaction_status = process_conversation_with_memory(
                            "I confirm this transaction",
                            st.session_state.memory,
                            client,
                            process_query_func,
                            get_exchange_rate_func
                        )
                        st.session_state.memory = updated_memory
                        st.experimental_rerun()
            
            with col_b:
                if st.button("Cancel"):
                    with st.spinner("Cancelling transaction..."):
                        # Process cancellation as a user message
                        response, updated_memory, transaction_status = process_conversation_with_memory(
                            "Cancel this transaction",
                            st.session_state.memory,
                            client,
                            process_query_func,
                            get_exchange_rate_func
                        )
                        st.session_state.memory = updated_memory
                        st.experimental_rerun()
        else:
            # Display current exchange rates
            st.subheader("Current Exchange Rates")
            
            try:
                usdt_aed = get_exchange_rate_func("USDT", "AED")
                usdt_irr = get_exchange_rate_func("USDT", "IRR")
                aed_irr = get_exchange_rate_func("AED", "IRR")
                
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
        
        # Display conversation summary and entities (could be collapsed/expandable in a real app)
        if st.session_state.memory.summary:
            st.subheader("Conversation Summary")
            st.markdown(st.session_state.memory.summary)
        
        # Show follow-up suggestions if available
        if st.session_state.memory.follow_up_questions:
            st.subheader("Suggested Questions")
            for question in st.session_state.memory.follow_up_questions:
                if st.button(question[:30] + "..." if len(question) > 30 else question):
                    # Process the suggested question as a user message
                    response, updated_memory, transaction_status = process_conversation_with_memory(
                        question,
                        st.session_state.memory,
                        client,
                        process_query_func,
                        get_exchange_rate_func
                    )
                    st.session_state.memory = updated_memory
                    st.experimental_rerun()


# ===================================
# Integration Code
# ===================================

# This code shows how to integrate the memory pipeline with your existing code

"""
# Import your existing functions
from your_module import process_query, get_exchange_rate, client

# Initialize the app with memory
if __name__ == "__main__":
    main_with_memory(process_query, get_exchange_rate, client)
"""