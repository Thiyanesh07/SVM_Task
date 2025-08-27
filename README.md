# 📧 Email Spam Classification (SVM & Decision Trees)

This project is a **Machine Learning based Spam Email Classifier** built using **Support Vector Machine (SVM)** and **Decision Trees**.  
It detects whether an email is **Spam** or **Not Spam** based on features like word frequencies and email length.  
The project also includes a **Streamlit frontend** for easy interaction.

---

## 🚀 Features
- Preprocessing of email dataset  
- Model training using:
  - Support Vector Machine (RBF kernel)
  - Decision Trees (comparison purpose)  
- Cross-validation with Accuracy & ROC-AUC metrics  
- Streamlit Web App for real-time prediction  
- Probability score for spam likelihood  

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **Scikit-learn**
- **Pandas, NumPy**
- **Streamlit**
- **Matplotlib/Seaborn** (for visualization)

---

## 📂 Project Structure

📁 SVM_Decision-Trees
│── app.py # Streamlit frontend
│── task.ipynb # Jupyter notebook (EDA + Model training)
│── email_spam_svm.csv # Dataset
│── requirements.txt # Dependencies
│── README.md # Project documentation


## ⚡ Setup Instructions

### 1️⃣ Clone the repository

```
git clone https://github.com/Thiyanesh07/SVM_Task.git
cd SVM_Task
```
### 2️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app
```
streamlit run app.py
```

### 🎯 Usage

Enter values for:

- word_freq_free

- word_freq_money

- word_freq_offer

- email_length

- Click Predict

- The app will display: Spam / Not Spam

 - Probability of being Spam

<img width="1259" height="787" alt="image" src="https://github.com/user-attachments/assets/e7a9104a-bd86-46d7-a7e3-afe3cb8fe1f4" />

### 📊 Model Performance

- Cross-Validation Accuracy: 0.97

- ROC-AUC Score: 0.99

### 🤝 Contributing

Pull requests are welcome! For major changes, open an issue first to discuss.



👨‍💻 Developed by Thiyanesh07
