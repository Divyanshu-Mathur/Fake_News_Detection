import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load model and vectorizer
with (open('model.pkl','rb')) as file:
    model=pickle.load(file)
with (open('vectorizer.pkl','rb')) as file:
    vec=pickle.load(file)

stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = text.lower() #lowering the text
    text = re.sub(r'[^a-z0-9\s]', '', text) #removing special characters
    text = re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , text) #removing hhtps
    tokens = word_tokenize(text)
    text = " ".join([word for word in tokens if word not in stop_words])
    return text

def predict(text):
    text = preprocess(text)
    vector = vec.transform([text])
    pred = model.predict(vector)
    return pred[0]

st.title('Fake News Detection')
input = st.text_input('Enter the news')
if st.button('Predict'):
    prediction = predict(input)
    if prediction == 0:
            st.error("Prediction: **Fake News**")
    else:
            st.success("Prediction: **Real News**")
