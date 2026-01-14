💬 Product FAQ Chatbot
A smart FAQ-based chatbot built using Streamlit, NLP (NLTK), and Machine Learning (TF-IDF + Cosine Similarity).
This chatbot answers user questions by matching them with the most relevant FAQ from a dataset.

🚀 Features
🧠 NLP-based question understanding (tokenization, stopword removal, lemmatization)
📊 TF-IDF Vectorization for text representation
📐 Cosine Similarity for best question matching
💬 Interactive chat UI using Streamlit
🌓 Dark-themed modern chat interface
📁 CSV-based FAQ dataset (easy to update)
🗂 Chat history using Streamlit session state
📧 Smart responses for greetings, thanks, and support queries

🛠️ Tech Stack
Python
Streamlit
NLTK
Scikit-learn
Pandas

📊 Dataset Format (faq_dataset.csv)

The CSV file should contain two columns:

question	answer
What is your return policy?	You can return products within 7 days.
How can I contact support?	Email us at support@example.com
⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/faq-chatbot.git
cd faq-chatbot

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Application
streamlit run app.py

📦 Required Python Packages
streamlit
pandas
nltk
scikit-learn

⚠️ NLTK resources are automatically downloaded on first run.
💡 How It Works
User enters a question
Text is preprocessed:
Punctuation removal
Tokenization
Stopword removal
Lemmatization
Questions are vectorized using TF-IDF
Cosine similarity finds the closest FAQ
Best matching answer is returned to the user

🎯 Example Use Cases
Product support bots
College or university FAQ assistants
Customer service automation
Internal company helpdesks

📸 UI Preview
Dark-themed chat interface with:
User messages on the right
Bot responses on the left
Scrollable chat history

⭐ Support
If you find this project helpful, give it a star ⭐ and feel free to contribute!
