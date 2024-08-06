import pickle
import streamlit as st

# Load the spam model and vectorizer
spam_model = pickle.load(open('/content/Spam_Model.sav', 'rb'))
vectorizer = pickle.load(open('/content/Vectorizer.sav', 'rb'))

st.title("Spam Message Detection Web App")
st.write("Enter a message to detect if it's spam or not")

user_message = st.text_area("Enter your message here", value='')

if st.button("Detect Spam"):
    # Preprocess and vectorize the user input
    user_message_vectorized = vectorizer.transform([user_message])
    
    # Predict spam probability
    spam_prediction = spam_model.predict(user_message_vectorized)
    
    if spam_prediction[0] == 1:
        spam_diagnosis = "The message is classified as spam"
    else:
        spam_diagnosis = "The message is not classified as spam"
    
    st.success(spam_diagnosis)