
import streamlit as st
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Customer Service Chatbot",
    page_icon="💬",
    layout="centered"
)

# -----------------------------
# Customer service knowledge base
# -----------------------------
data = {
    "greeting": {
        "questions": [
            "hello", "hi", "hey", "good morning",
            "good afternoon", "how are you"
        ],
        "responses": [
            "Hello! 👋 Welcome to Customer Support. How can I help you today?",
            "Hi there! How may I assist you?"
        ]
    },
    "order_status": {
        "questions": [
            "where is my order",
            "track my order",
            "order status",
            "when will my order arrive",
            "check delivery status"
        ],
        "responses": [
            "You can check your order status in the Orders section "
            "of your account. Please note that I cannot access live "
            "order records in this demo."
        ]
    },
    "returns": {
        "questions": [
            "how can I return my product",
            "return an item",
            "what is your return policy",
            "I want to return my order",
            "how do I send a product back"
        ],
        "responses": [
            "To request a return, open your Orders section, select "
            "the item, and check the available return options. "
            "Eligibility depends on the store's actual return policy."
        ]
    },
    "refund": {
        "questions": [
            "where is my refund",
            "how long does a refund take",
            "refund status",
            "I want my money back",
            "when will I receive my refund"
        ],
        "responses": [
            "Refund timelines depend on the store and payment method. "
            "Check your refund confirmation or contact the support team "
            "for the actual status of your refund."
        ]
    },
    "payment": {
        "questions": [
            "payment failed",
            "money deducted but order not placed",
            "payment methods",
            "how can I pay",
            "my transaction failed"
        ],
        "responses": [
            "If money was deducted but your order was not placed, "
            "check your transaction history and contact your bank "
            "or the merchant's support team if the issue continues."
        ]
    },
    "delivery": {
        "questions": [
            "delivery charges",
            "how many days for delivery",
            "shipping time",
            "when will my package arrive",
            "is delivery free"
        ],
        "responses": [
            "Delivery times and shipping charges depend on the "
            "delivery address and selected service. Please check "
            "the delivery information at checkout."
        ]
    },
    "contact": {
        "questions": [
            "contact customer support",
            "how can I speak to an agent",
            "customer care number",
            "talk to a human",
            "email support"
        ],
        "responses": [
            "You can contact the company's official support team "
            "through its verified website or help center. "
            "This demo does not have a live human-agent connection."
        ]
    },
    "thanks": {
        "questions": [
            "thank you", "thanks", "thank you so much",
            "that helped", "great thanks"
        ],
        "responses": [
            "You're welcome! 😊 Is there anything else I can help with?",
            "Happy to help! Let me know if you have another question."
        ]
    },
    "goodbye": {
        "questions": [
            "bye", "goodbye", "see you later",
            "that is all", "exit"
        ],
        "responses": [
            "Thank you for chatting with us! Have a great day. 👋",
            "Goodbye! Feel free to return if you need help."
        ]
    }
}

# -----------------------------
# Prepare NLP training examples
# -----------------------------
training_questions = []
training_intents = []

for intent, details in data.items():
    for question in details["questions"]:
        training_questions.append(question)
        training_intents.append(intent)

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    strip_accents="unicode"
)

question_vectors = vectorizer.fit_transform(training_questions)


def get_response(user_message):
    cleaned_message = re.sub(
        r"[^a-zA-Z0-9\s]", " ", user_message.lower()
    ).strip()

    if not cleaned_message:
        return "Please enter a question so I can help you.", 0.0

    message_vector = vectorizer.transform([cleaned_message])
    similarities = cosine_similarity(
        message_vector, question_vectors
    )[0]

    best_index = similarities.argmax()
    confidence = float(similarities[best_index])
    intent = training_intents[best_index]

    # Avoid misleading answers for unrelated questions
    if confidence < 0.18:
        return (
            "Sorry, I couldn't understand that question. "
            "Please try asking about orders, returns, refunds, "
            "payments, delivery, or customer support.",
            confidence
        )

    response = random.choice(data[intent]["responses"])
    return response, confidence


# -----------------------------
# Streamlit interface
# -----------------------------
st.title("💬 Customer Service Chatbot")
st.caption("Your virtual assistant for everyday customer queries")

with st.sidebar:
    st.header("About this project")
    st.write(
        "This chatbot uses TF-IDF and cosine similarity "
        "to identify customer questions and provide responses."
    )

    st.subheader("Topics supported")
    st.write(
        "• Order tracking\n"
        "• Returns and refunds\n"
        "• Payment problems\n"
        "• Delivery information\n"
        "• Contact support"
    )

    if st.button("Clear conversation"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! 👋 Welcome to Customer Support. "
                    "How can I help you today?"
                )
            }
        ]
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 Welcome to Customer Support. "
                "Ask me about orders, returns, refunds, or payments."
            )
        }
    ]

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Suggested questions
st.subheader("Try asking")

suggestions = [
    "Where is my order?",
    "How do I get a refund?",
    "My payment failed",
    "How can I return an item?"
]

cols = st.columns(2)

for i, suggestion in enumerate(suggestions):
    if cols[i % 2].button(suggestion, key=f"suggest_{i}"):
        st.session_state.pending_question = suggestion

prompt = st.chat_input("Type your question here...")

if prompt:
    st.session_state.pending_question = prompt

if st.session_state.get("pending_question"):
    user_input = st.session_state.pending_question
    del st.session_state.pending_question

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    response, confidence = get_response(user_input)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    st.rerun()
