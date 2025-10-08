import streamlit as st
import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')




    df = pd.read_csv("faq_dataset.csv")

    def preprocessing(text):
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word.lower() not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word, pos="v") for word in tokens]
        return " ".join(tokens)

    df["clean_question"] = df["question"].apply(preprocessing)
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(df["clean_question"])
    return df, vectorizer, x

df, vectorizer, x = load_data()

def faq_chatbot(user_input):
    def preprocessing(text):
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word.lower() not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word, pos="v") for word in tokens]
        return " ".join(tokens)

    cleaned_input = preprocessing(user_input)
    user_vector = vectorizer.transform([cleaned_input])
    similarity = cosine_similarity(x, user_vector)
    index = similarity.argmax()
    return df["answer"].iloc[index]

st.set_page_config(page_title="FAQ Chatbot", page_icon="💬", layout="centered")

st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: white;
        }
        .chat-bubble-user {
            background-color: #0056ff;
            color: white;
            padding: 10px 15px;
            border-radius: 20px;
            text-align: right;
            margin: 5px 0px 5px 50px;
        }
        .chat-bubble-bot {
            background-color: #2c2c2c;
            color: white;
            padding: 10px 15px;
            border-radius: 20px;
            text-align: left;
            margin: 5px 50px 5px 0px;
        }
        .chat-box {
            background-color: #1e1e1e;
            padding: 20px;
            border-radius: 10px;
            max-height: 500px;
            overflow-y: auto;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💬 Product FAQ Chatbot")
st.caption("Ask me anything about our products — I’m here to help!")

# Session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.history.append({"sender": "user", "message": user_input})

    if any(word in user_input.lower() for word in ["help", "support", "email", "contact", "number", "call"]):
        response = "Sure! Please wait — our support team will reach out soon at -> support@example.com 📧"
    elif user_input.lower() in ["okay","ok"]:
        response="If you want further detail contact us at -> support@example.com"
    elif user_input.lower() in ["hi","hey","hello","hi sir", "hello sir","hey sir"]:
        response=f"{user_input.capitalize()}, i'm product assistence how can i help you."
    elif user_input.lower() in ["thank you","thanks","tnx"]:
        response="Your welcome.If you want further detail contact us at -> support@example.com 📧"
    else:
        response = faq_chatbot(user_input)

    st.session_state.history.append({"sender": "bot", "message": response})

st.markdown('<div class="chat-box">', unsafe_allow_html=True)
for chat in st.session_state.history:
    if chat["sender"] == "user":
        st.markdown(f"<div class='chat-bubble-user'>{chat['message']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-bot'>{chat['message']}</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
