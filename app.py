import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import streamlit as st

data = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\spam\spam.csv", encoding='latin-1')[['Category', 'Message']]
data.drop_duplicates(inplace=True)
data.dropna(subset=['Message'], inplace=True)  # 💥 Drop rows with NaN messages
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])


mess = data['Message']
catg = data['Category']
mess_train, mess_test, catg_train, catg_test = train_test_split(mess, catg, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english')
features_train = vectorizer.fit_transform(mess_train)
features_test = vectorizer.transform(mess_test)

model = MultinomialNB()
model.fit(features_train, catg_train)


accuracy = accuracy_score(catg_test, model.predict(features_test))


def predict(message):
    input_vector = vectorizer.transform([message])
    result = model.predict(input_vector)
    return result[0]


st.set_page_config(page_title="Spam Detector", layout="centered")
st.markdown("<h1 style='text-align: center;'>📩 Spam Detection App</h1>", unsafe_allow_html=True)
st.markdown("This app predicts whether a message is **Spam** or **Not Spam** using a machine learning model.")

input_message = st.text_input('✉️ Enter your message below:')

if st.button('🔍 Check'):
    if input_message.strip() == "":
        st.warning("Please enter a message before validating.")
    else:
        prediction = predict(input_message)
        if prediction == "Spam":
            st.error("🚨 This message is **SPAM**!")
        else:
            st.success("✅ This message is **NOT SPAM**.")

st.markdown(f"<br><sub>📊 Model Accuracy: <b>{accuracy * 100:.2f}%</b></sub>", unsafe_allow_html=True)






