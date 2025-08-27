import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

df = pd.read_csv("email_spam_svm.csv")

x = df[['word_freq_free','word_freq_money','word_freq_offer','email_length']]
y = df['is_spam']



pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", probability=True, random_state=42))
])
pipeline.fit(x, y)

cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)



st.set_page_config(page_title="Spam Classifier", layout="centered")

st.title("📧 Email Spam Classifier (SVM)")
st.write("Enter the email statistics below to check if it’s **Spam** or **Not Spam**.")


word_freq_free = st.number_input("Word Frequency: 'free' (%)", min_value=0.0, max_value=100.0, value=2.1, step=0.1)
word_freq_money = st.number_input("Word Frequency: 'money' (%)", min_value=0.0, max_value=100.0, value=1.3, step=0.1)
word_freq_offer = st.number_input("Word Frequency: 'offer' (%)", min_value=0.0, max_value=100.0, value=0.7, step=0.1)
email_length = st.number_input("Email Length (characters)", min_value=10, max_value=5000, value=180, step=10)


if st.button("🔍 Predict"):
    new_email = pd.DataFrame([[word_freq_free, word_freq_money, word_freq_offer, email_length]],
                             columns=['word_freq_free','word_freq_money','word_freq_offer','email_length'])
    
    prediction = pipeline.predict(new_email)[0]
    probability = pipeline.predict_proba(new_email)[0][1]

    st.subheader("📊 Result")
    if prediction == 1:
        st.error(f"🚨 This email is classified as **Spam** with probability {probability:.2f}")
    else:
        st.success(f"✅ This email is classified as **Not Spam** with probability {1 - probability:.2f}")
